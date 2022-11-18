import os
import utils
import json
import argparse
import pytorch_lightning as pl
import torch as th
import numpy as np
from sklearn.metrics import confusion_matrix
import random

proj_name = 'Food-Fixed-Seed'
check_interval = 10

def linear_probe_sklearn_main(cfg):
    "Adapted from: https://github.com/KaiyangZhou/CoOp/blob/main/lpclip/linear_probe.py"
    from sklearn.linear_model import LogisticRegression
    from models.linear_probe.linear_probe import get_features
    from data_lp import LinearProbeDataModule

    val_acc_step_list = np.zeros([cfg.n_runs, cfg.steps])
    best_c_weights_list = []

    for seed in range(1, cfg.n_runs + 1):
        np.random.seed(seed)
        random.seed(seed)
        data_module = LinearProbeDataModule(cfg.data_root,
                                            cfg.bs,
                                            cfg.img_split_path,
                                            cfg.img_path,
                                            cfg.n_shots,
                                            cfg.cls_names,
                                            img_ext=cfg.img_ext,
                                            num_workers=8)
        data_module.setup()
        train_features, train_labels = get_features(data_module.train_dataloader(),
                                                    cfg.paper, cfg.clip_model)
        val_features, val_labels = get_features(data_module.val_dataloader(),
                                                cfg.paper, cfg.clip_model)

        # search initialization
        search_list = [1e6, 1e4, 1e2, 1, 1e-2, 1e-4, 1e-6]
        acc_list = []
        for c_weight in search_list:
            clf = LogisticRegression(solver="lbfgs", max_iter=1000, penalty="l2", C=c_weight).fit(train_features, train_labels)
            pred = clf.predict(val_features)
            acc_val = np.mean([int(t==p) for t,p in zip(val_labels, pred)]).astype(np.float32) * 100.
            acc_list.append(acc_val)

        print(acc_list, flush=True)

        # binary search
        peak_idx = np.argmax(acc_list)
        c_peak = search_list[peak_idx]
        c_left, c_right = 1e-1 * c_peak, 1e1 * c_peak

        def binary_search(c_left, c_right, seed, step, val_acc_step_list):
            clf_left = LogisticRegression(#random_state=0,
                                            C=c_left,
                                            max_iter=1000,
                                            verbose=1,
                                            n_jobs=8)
            clf_left.fit(train_features, train_labels)
            pred_left = clf_left.predict(val_features)
            accuracy = np.mean((val_labels == pred_left).astype(np.float32)) * 100.
            acc_left =  np.mean([int(t==p) for t,p in zip(val_labels, pred_left)]).astype(np.float32) * 100
            print("Val accuracy (Left): {:.2f}".format(acc_left), flush=True)

            clf_right = LogisticRegression(solver="lbfgs", max_iter=1000, penalty="l2", C=c_right).fit(train_features, train_labels)
            pred_right = clf_right.predict(val_features)
            acc_right =  np.mean( [int(t==p) for t,p in zip(val_labels, pred_right)]).astype(np.float32) * 100
            print("Val accuracy (Right): {:.2f}".format(acc_right), flush=True)

            # find maximum and update ranges
            if acc_left < acc_right:
                c_final = c_right
                clf_final = clf_right
                # range for the next step
                c_left = 0.5 * (np.log10(c_right) + np.log10(c_left))
                c_right = np.log10(c_right)
            else:
                c_final = c_left
                clf_final = clf_left
                # range for the next step
                c_right = 0.5 * (np.log10(c_right) + np.log10(c_left))
                c_left = np.log10(c_left)

            pred = clf_final.predict(val_features)
            val_acc = np.mean( [int(t==p) for t,p in zip(val_labels, pred)]).astype(np.float32) * 100
            print("Val Accuracy: {:.2f}".format(val_acc), flush=True)
            val_acc_step_list[seed - 1, step] = val_acc

            saveline = "{}, seed {}, {} shot, weight {}, val_acc {:.2f}\n".format(cfg.dataset, seed, cfg.n_shots, c_final, val_acc)
            return (
                np.power(10, c_left),
                np.power(10, c_right),
                seed,
                step,
                val_acc_step_list,
            )

        for step in range(cfg.steps):
            print(
                f"{cfg.dataset}, {cfg.n_shots} Shot, Round {step}: {c_left}/{c_right}",
                flush=True,
            )
            c_left, c_right, seed, step, val_acc_step_list = binary_search(c_left, c_right, seed, step, val_acc_step_list)

        # save c_left as the optimal weight for each run
        best_c_weights_list.append(c_left)

    # take the mean of C from multiple runs
    best_c = np.mean(best_c_weights_list)
    
    # run test with the best C
    classifier = LogisticRegression(#random_state=0,
                                    C=best_c,
                                    max_iter=1000,
                                    verbose=1,
                                    n_jobs=8)
    classifier.fit(train_features, train_labels)
    test_features, test_labels = get_features(data_module.test_dataloader(),
                                              cfg.paper, cfg.clip_model)
    predictions = classifier.predict(test_features)

    # average validation performance
    val_acc_list = val_acc_step_list[:, -1]
    val_acc_mean = np.mean(val_acc_list)
    val_acc_std = np.std(val_acc_list)
    print("{} {}shot {}, best C = {}\n".format(cfg.dataset, cfg.n_shots, cfg.clip_model, best_c))
    print(f"Val Accuracy: {val_acc_mean:.3f}")
    print(f"Val std: {val_acc_std:.3f}")

    # test performance
    accuracy = np.mean((test_labels == predictions).astype(np.float32)) * 100.
    print(f"Test Accuracy = {accuracy:.3f}")

    with open("output/linear_probe/{}.txt".format(cfg.dataset), "a") as f:
        f.write("{} {}shot {}, best C = {}, seed = {}\n".format(cfg.dataset, cfg.n_shots, cfg.clip_model, best_c, seed))
        f.write(f"Val Accuracy: {val_acc_mean:.3f}, Val std: {val_acc_std:.3f}\n")
        f.write(f"Test Accuracy = {accuracy:.3f}\n\n")
    matrix = confusion_matrix(test_labels, predictions)
    print("Macro Accuracy = {}".format(np.mean(matrix.diagonal()/matrix.sum(axis=1)) * 100))

def save_npy_files(class2concepts, save_dir):
    # sort class name to make sure they are in the same order, to avoid potential problem
    class_names = sorted(list(class2concepts.keys()))
    num_concept = sum([len(concepts) for concepts in class2concepts.values()])
    concept2cls = np.zeros(num_concept)
    i = 0
    all_concepts = []
    for class_name, concepts in class2concepts.items():
        class_idx = class_names.index(class_name)
        for concept in concepts:
            all_concepts.append(concept)
            concept2cls[i] = class_idx
            i += 1
    np.save(save_dir + 'concepts_raw.npy', np.array(all_concepts))
    np.save(save_dir + 'cls_names.npy', np.array(class_names))
    np.save(save_dir + 'concept2cls.npy', concept2cls)


def asso_opt_main(cfg):
    from models.asso_opt.asso_opt import AssoConcept, AssoConceptFast
    from models.select_concept.select_algo import mi_select, clip_score_select, group_mi_select, group_clip_select, submodular_select, random_select
    from data import DataModule, DotProductDataModule
    import random
    proj_name = cfg.proj_name

    if not os.path.isfile(cfg.cls_name_path):
        class2concepts = json.load(open(cfg.concept_root + "class2concepts.json", "r"))
        save_npy_files(class2concepts, cfg.concept_root)
    
    # concept seletion method
    if cfg.concept_select_fn == "submodular":
        concept_select_fn = submodular_select
        print("use submodular")
    elif cfg.concept_select_fn == "random":
        concept_select_fn = random_select
        print("use random")
    else:
        if cfg.use_mi:
            if cfg.group_select:
                concept_select_fn = group_mi_select
                print("use grounp mi")
            else:
                concept_select_fn = mi_select
                print("use mi")
        else:
            if cfg.group_select:
                concept_select_fn = group_clip_select
                print("use group clip")
            else:
                concept_select_fn = clip_score_select
                print("use clip")
    
    random.seed(1) # seed matches first run of linear probe

    try: print(cfg.submodular_weights)
    except: cfg.submodular_weights = "none"
    if cfg.proj_name == "ImageNet" and (cfg.n_shots == "all" or cfg.n_shots == 16):
        print("use image feature dataloader")
        data_module = DataModule(
            cfg.num_concept,
            cfg.data_root,
            cfg.clip_model,
            cfg.img_split_path,
            cfg.img_path,
            cfg.n_shots,
            cfg.raw_sen_path,
            cfg.concept2cls_path,
            concept_select_fn,
            cfg.cls_name_path,
            cfg.bs,
            on_gpu=cfg.on_gpu,
            num_workers=cfg.num_workers if 'num_workers' in cfg else 0,
            img_ext=cfg.img_ext if 'img_ext' in cfg else '.jpg',
            clip_ckpt=cfg.ckpt_path if 'ckpt_path' in cfg else None,
            use_txt_norm=cfg.use_txt_norm if 'use_txt_norm' in cfg else False, 
            use_img_norm=cfg.use_img_norm if 'use_img_norm' in cfg else False,
            use_cls_name_init=cfg.cls_name_init if 'cls_name_init' in cfg else 'none',
            use_cls_sim_prior=cfg.cls_sim_prior if 'cls_sim_prior' in cfg else 'none',
            remove_cls_name=cfg.remove_cls_name if 'remove_cls_name' in cfg else True,
            submodular_weights=cfg.submodular_weights
            )
    
    else:
        print("use dot product dataloader")
        data_module = DotProductDataModule(
            cfg.num_concept,
            cfg.data_root,
            cfg.clip_model,
            cfg.img_split_path,
            cfg.img_path,
            cfg.n_shots,
            cfg.raw_sen_path,
            cfg.concept2cls_path,
            concept_select_fn,
            cfg.cls_name_path,
            cfg.bs,
            on_gpu=cfg.on_gpu,
            num_workers=cfg.num_workers if 'num_workers' in cfg else 0,
            img_ext=cfg.img_ext if 'img_ext' in cfg else '.jpg',
            clip_ckpt=cfg.ckpt_path if 'ckpt_path' in cfg else None,
            use_txt_norm=cfg.use_txt_norm if 'use_txt_norm' in cfg else False, 
            use_img_norm=cfg.use_img_norm if 'use_img_norm' in cfg else False,
            use_cls_name_init=cfg.cls_name_init if 'cls_name_init' in cfg else 'none',
            use_cls_sim_prior=cfg.cls_sim_prior if 'cls_sim_prior' in cfg else 'none',
            remove_cls_name=cfg.remove_cls_name if 'remove_cls_name' in cfg else True,
            submodular_weights=cfg.submodular_weights
            )

    if cfg.test:
        ckpt_path = cfg.ckpt_path
        print('load ckpt: {}'.format(ckpt_path))
        model = AssoConceptFast.load_from_checkpoint(str(ckpt_path))
        trainer = pl.Trainer(gpus=1)
        trainer.test(model, data_module)
        test_acc = round(100 * float(model.total_test_acc), 2)
        dataset = cfg.ckpt_path.split("/")[-3]
        exp = cfg.ckpt_path.split("/")[-2]
        with open("output/asso_opt/{}.txt".format(dataset), "a") as f:
            f.write("{}\t{}\n".format(exp, test_acc))
        return

    if cfg.proj_name == "ImageNet" and (cfg.n_shots == "all" or cfg.n_shots == 16):
            print("use asso concept with image feature loader")
            model = AssoConcept(cfg, init_weight=th.load(cfg.init_weight_path) if 'init_weight_path' in cfg else None)
    else:
        print("use asso concept with dot product loader, faster")
        model = AssoConceptFast(cfg, init_weight=th.load(cfg.init_weight_path) if 'init_weight_path' in cfg else None)

    if cfg.proj_name == "ImageNet" and cfg.n_shots == "all": check_interval = 5
    else: check_interval = 10

    print("check interval = {}".format(check_interval))

    if not cfg.DEBUG:
        if 'use_last_ckpt' in cfg and cfg['use_last_ckpt']:
            checkpoint_callback = pl.callbacks.ModelCheckpoint(
                dirpath=cfg.work_dir,
                filename='{epoch}-{step}-{val_acc:.4f}',
                every_n_epochs=check_interval)
        else:
            checkpoint_callback = pl.callbacks.ModelCheckpoint(
                dirpath=cfg.work_dir,
                filename='{epoch}-{step}-{val_acc:.4f}',
                monitor='val_acc',
                mode='max',
                save_top_k=1,
                every_n_epochs=check_interval)

        wandb_logger = pl.loggers.WandbLogger(name='{}shot_{}_{}_{}'\
                       .format(cfg.n_shots, cfg.concept_type, cfg.num_concept, cfg.submodular_weights[1]),
                       project=proj_name,
                       config=cfg._cfg_dict)

        trainer = pl.Trainer(gpus=1,
                             callbacks=[checkpoint_callback],
                             logger=wandb_logger,
                             check_val_every_n_epoch=check_interval,
                             max_epochs=cfg.max_epochs if 'max_epochs' in cfg else 1000)
    else:
        if 'use_last_ckpt' in cfg and cfg['use_last_ckpt']:
            checkpoint_callback = pl.callbacks.ModelCheckpoint(
                dirpath=cfg.work_dir,
                filename='{epoch}-{step}-{val_acc:.4f}',
                every_n_epochs=check_interval)

        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            dirpath=cfg.work_dir,
            filename='{epoch}-{step}-{val_acc:.4f}',
            monitor='val_acc',
            mode='max',
            save_top_k=1,
            every_n_epochs=50)
        # device_stats = pl.callbacks.DeviceStatsMonitor()
        trainer = pl.Trainer(gpus=1, max_epochs=1000, \
            callbacks=[checkpoint_callback, ], check_val_every_n_epoch=50, default_root_dir=cfg.work_dir)
    trainer.fit(model, data_module)


if __name__ == "__main__":
    from mmcv import DictAction
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', help='path to config file')
    parser.add_argument('--work-dir',
                        help='work directory to save the ckpt and config file')
    parser.add_argument('--func', help='which task to run')
    parser.add_argument('--test',
                        action='store_true',
                        help='whether to enable test mode')
    parser.add_argument('--DEBUG',
                        action='store_true',
                        help='whether to enable debug mode')
    parser.add_argument('--cfg-options',
                        nargs='+',
                        action=DictAction,
                        help='overwrite parameters in cfg from commandline')
    args = parser.parse_args()
    if not args.test:
        cfg = utils.pre_exp(args.cfg, args.work_dir)
    else:
        from mmcv import Config
        cfg = Config.fromfile(args.cfg)
    cfg.test = args.test
    cfg.DEBUG = args.DEBUG
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    main = eval(args.func)
    main(cfg)