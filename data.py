"""
This is a cleaned version of data loader for new interfaces; 
The datamodule here handles all data processing including concept selection.
For the model, it only loads data processed and save in data_root.
"""
import torch as th
import numpy as np
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pathlib import Path
import random
import utils


class ImageFeatDataset(Dataset):
    """
    Provide (image, label) pair for association matrix optimization,
    where image is a PIL Image
    """

    def __init__(self, img_feat, label, on_gpu):
        self.img_feat = img_feat.cuda() if on_gpu else img_feat
        self.labels = label.cuda() if on_gpu else label

    def __len__(self):
        return len(self.img_feat)

    def __getitem__(self, idx):
        return self.img_feat[idx], self.labels[idx]


class DotProductDataset(Dataset):
    """
    Provide (image, label) pair for association matrix optimization,
    where image is a PIL Image
    """

    def __init__(self, img_feat, txt_feat, label, on_gpu):
        self.dot_product = (img_feat @ txt_feat.t())
        self.dot_product = self.dot_product.cuda(
        ) if on_gpu else self.dot_product
        self.labels = label.cuda() if on_gpu else label
        # uncomment for imagenet all shot
        # self.dot_product = (img_feat @ txt_feat.t())
        # self.labels = label


    def __len__(self):
        return len(self.dot_product)


    def __getitem__(self, idx):
        # return self.dot_product[idx].cuda(), self.labels[idx].cuda()
        return self.dot_product[idx], self.labels[idx]


class Dataset_with_name(Dataset):
    def __init__(self, ori_dataset, names):
        assert len(ori_dataset) == len(names)
        self.names = names 
        self.ori_dataset = ori_dataset 
    
    
    def __len__(self): 
        return len(self.ori_dataset)
    

    def __getitem__(self, idx):
        return self.ori_dataset[idx] + (str(self.names[idx]), )


class DataModule(pl.LightningDataModule):
    """
    It prepares image and concept CLIP features given config of one dataset.
    """
    def __init__(
            self,
            num_concept,
            data_root,
            clip_model,
            img_split_path,
            img_root,
            n_shots,
            concept_raw_path, 
            concept2cls_path, 
            concept_select_fn, 
            cls_names_path,
            batch_size,
            use_txt_norm=False,
            use_img_norm=False,
            num_workers=0,
            img_ext='.jpg',
            clip_ckpt=None,
            on_gpu=False,
            force_compute=True,
            use_cls_name_init='none',
            use_cls_sim_prior='none',
            remove_cls_name=False,
            submodular_weights=None):
        super().__init__()
        
        # image feature is costly to compute, so it will always be cached
        self.force_compute = force_compute 
        self.use_txt_norm = use_txt_norm 
        self.use_img_norm = use_img_norm
        self.use_cls_name_init = use_cls_name_init
        self.use_cls_sim_prior = use_cls_sim_prior
        self.remove_cls_name = remove_cls_name
        self.data_root = Path(data_root)
        self.data_root.mkdir(exist_ok=True)
        self.img_split_path = Path(img_split_path)
        self.img_split_path.mkdir(exist_ok=True, parents=True)

        # all variables save_dir that will be created inside this module
        self.img_feat_save_dir = {
            mode: self.img_split_path.joinpath(
                'img_feat_{}_{}_{}{}_{}.pth'.format(mode, n_shots, int(self.use_img_norm), int(self.use_txt_norm), clip_model.replace('/','-')) if mode ==
                'train' else 'img_feat_{}_{}{}_{}.pth'.format(mode, int(self.use_img_norm), int(self.use_txt_norm), clip_model.replace('/', '-')))
            for mode in ['train', 'val', 'test']
        }
        self.label_save_dir = {
            mode: self.img_split_path.joinpath(
                'label_{}_{}.pth'.format(mode, n_shots) if mode ==
                'train' else 'label_{}.pth'.format(mode))
            for mode in ['train', 'val', 'test']
        }
        if self.use_cls_name_init != 'none':
            self.init_weight_save_dir = self.data_root.joinpath('init_weight.pth')
        if self.use_cls_sim_prior != 'none':
            self.cls_sim_save_dir = self.data_root.joinpath('cls_sim.pth')
        self.select_idx_save_dir = self.data_root.joinpath(
            'select_idx.pth')  # selected concept indices
        self.concepts_raw_save_dir = self.data_root.joinpath(
            'concepts_raw_selected.npy')
        self.concept2cls_save_dir = self.data_root.joinpath(
            'concept2cls_selected.npy')
        self.concept_feat_save_dir = self.data_root.joinpath(
            'concepts_feat_{}.pth'.format(clip_model.replace('/','-')))

        self.clip_model = clip_model
        self.clip_ckpt = clip_ckpt
        self.cls_names = np.load(cls_names_path).tolist() # for reference, the mapping between indices and names
        self.num_concept = num_concept
        self.submodular_weights = submodular_weights

        # handling image related data
        self.splits = {
            split: utils.pickle_load(
                self.img_split_path.joinpath(
                    'class2images_{}.p'.format(split)))
            for split in ['train', 'val', 'test']
        }

        self.n_shots = n_shots
        self.img_root = Path(img_root)
        self.img_ext = img_ext
        self.prepare_img_feat(self.splits, self.n_shots, self.clip_model, self.clip_ckpt)

        if self.n_shots != "all": 
            self.num_images_per_class = [self.n_shots] * len(self.splits['train'])
        else:
            self.num_images_per_class = [len(images) for _, images in self.splits['train'].items()]

        # handling concept related data
        self.concepts_raw = np.load(concept_raw_path)
        self.concept2cls = np.load(concept2cls_path)
        
        # TODO: remove duplication
        self.concepts_raw, idx = self.preprocess(self.concepts_raw, self.cls_names)
        self.concept2cls = self.concept2cls[idx] 
        self.concept_select_fn = concept_select_fn

        if self.n_shots != "all":
            assert len(self.img_feat['train']) == len(self.cls_names) * self.n_shots

        self.prepare_txt_feat(self.concepts_raw, self.clip_model, self.clip_ckpt)

        self.select_concept(self.concept_select_fn, self.img_feat['train'], self.concept_feat, self.n_shots, self.num_concept, self.concept2cls, self.clip_ckpt, self.num_images_per_class, self.submodular_weights)

        # save all raw concepts and coresponding classes as a reference
        np.save(self.concepts_raw_save_dir, self.concepts_raw)
        np.save(self.concept2cls_save_dir, self.concept2cls)

        if self.use_cls_name_init != 'none':
            self.gen_init_weight_from_cls_name(self.cls_names, self.concepts_raw[self.select_idx])

        if self.use_cls_sim_prior != 'none':
            split = 'train'
            self.gen_mask_from_img_sim(self.img_feat[split], self.n_shots, self.label[split][::self.n_shots])

        # parameters for dataloader
        self.bs = batch_size
        self.num_workers = num_workers
        self.on_gpu = on_gpu


    def check_pattern(self, concepts, pattern):
        """
        Return a boolean array where it is true if one concept contains the pattern 
        """
        return np.char.find(concepts, pattern) != -1


    def check_no_cls_names(self, concepts, cls_names):
        res = np.ones(len(concepts), dtype=bool)
        for cls_name in cls_names: 
            no_cls_name = ~self.check_pattern(concepts, cls_name)
            res = res & no_cls_name 
        return res


    def preprocess(self, concepts, cls_names=None):
        """
        concepts: numpy array of strings of concepts
        
        This function checks all input concepts, remove duplication, and 
        remove class names if necessary
        """
        concepts, left_idx = np.unique(concepts, return_index=True)
        if self.remove_cls_name: 
            print('remove cls name')
            is_good = self.check_no_cls_names(concepts, cls_names)
            concepts = concepts[is_good]
            left_idx = left_idx[is_good]
        return concepts, left_idx


    def gen_init_weight_from_cls_name(self, cls_names, concepts):
        # always use unnormalized text feature for more accurate class-concept assocation
        num_cls = len(cls_names)
        num_concept_per_cls = self.num_concept // num_cls
        cls_name_feat = utils.prepare_txt_feat(cls_names, clip_model_name=self.clip_model, ckpt_path=self.clip_ckpt)
        concept_feat = utils.prepare_txt_feat(concepts, clip_model_name=self.clip_model, ckpt_path=self.clip_ckpt)
        dis = th.cdist(cls_name_feat, concept_feat)
        # select top k concept with smallest distanct to the class name
        _, idx = th.topk(dis, num_concept_per_cls, largest=False)
        init_weight = th.zeros((num_cls, self.num_concept))
        init_weight.scatter_(1, idx, 1)
        th.save(init_weight, self.init_weight_save_dir)


    def gen_mask_from_img_sim(self, img_feat, n_shots, label):
        print('generate cls sim mask')
        num_cls = len(img_feat) // n_shots
        img_feat = img_feat / (img_feat.norm(dim=-1, keepdim=True) + 1e-7)
        img_sim = img_feat @ img_feat.T
        class_sim = th.empty((num_cls, num_cls))
        for i, row_split in enumerate(th.split(img_sim, n_shots, dim=0)):
            for j, col_split in enumerate(th.split(row_split, n_shots, dim=1)):
                class_sim[label[i], label[j]] = th.mean(col_split)

        good = class_sim >= th.quantile(class_sim, 0.95, dim=-1)
        final_sim = th.zeros(class_sim.shape)
        for i in range(num_cls):
            for j in range(num_cls):
                if i == j: final_sim[i, j] = 1
                elif good[i, j] == True: final_sim[i, j] = class_sim[i, j]

        th.save(final_sim, self.cls_sim_save_dir)
        self.class_sim = final_sim
    

    def select_concept(self, concept_select_fn, img_feat_train, concept_feat, n_shots, num_concepts, concept2cls, clip_ckpt, num_images_per_class, submodular_weights):
        if not self.select_idx_save_dir.exists() or (self.force_compute and not clip_ckpt):
            print('select concept')
            self.select_idx = concept_select_fn(img_feat_train, concept_feat, n_shots, concept2cls, 
                                                num_concepts, num_images_per_class, submodular_weights)
            th.save(self.select_idx, self.select_idx_save_dir)
        else:
            self.select_idx = th.load(self.select_idx_save_dir)


    def prepare_txt_feat(self, concepts_raw, clip_model, clip_ckpt):
        # TODO: it is possible to store a global text feature for all concepts
        # Here, we just be cautious to recompute it every time
        if not self.concept_feat_save_dir.exists() or self.force_compute:
            print('prepare txt feat')
            self.concept_feat = utils.prepare_txt_feat(concepts_raw,
                                                   clip_model_name=clip_model,
                                                   ckpt_path=None)
            th.save(self.concept_feat, self.concept_feat_save_dir)
            
        else:
            self.concept_feat = th.load(self.concept_feat_save_dir)

        if self.use_txt_norm:
            self.concept_feat /= self.concept_feat.norm(dim=-1, keepdim=True)


    def get_img_n_shot(self, cls2img, n_shots):
        labels = []
        all_img_paths = []
        for cls_name, img_names in cls2img.items():
            if n_shots != 'all': img_names = random.sample(img_names, n_shots) # random sample n shot images
            labels.extend([self.cls_names.index(cls_name)] * len(img_names))
            all_img_paths.extend([self.img_root.joinpath('{}{}'.format(img_name, self.img_ext)) for img_name in img_names])
        return all_img_paths, labels


    def compute_img_feat(self, cls2img, n_shots, clip_model, clip_ckpt):
        all_img_paths, labels = self.get_img_n_shot(cls2img, n_shots)
        img_feat = utils.prepare_img_feat(all_img_paths,
                                          clip_model_name=clip_model,
                                          ckpt_path=clip_ckpt)
        return img_feat, th.tensor(labels)


    def prepare_img_feat(self, splits, n_shots, clip_model, clip_ckpt):
        self.img_feat = {}
        self.label = {}
        for mode in ['train', 'val', 'test']:
            cls2img, feat_save_dir, label_save_dir = splits[mode], self.img_feat_save_dir[mode], self.label_save_dir[mode]

            if not feat_save_dir.exists():
                print('compute img feat for {}'.format(mode))
                img_feat, label = self.compute_img_feat(cls2img, n_shots if mode == 'train' else 'all', clip_model, clip_ckpt)
                th.save(img_feat, feat_save_dir)
                th.save(label, label_save_dir)
            else:
                img_feat, label = th.load(feat_save_dir), th.load(label_save_dir)
                
            if self.use_img_norm:
                img_feat /= img_feat.norm(dim=-1, keepdim=True)

            self.img_feat[mode] = img_feat
            self.label[mode] = label
            

    def setup(self, stage):
        """
        Set up datasets for dataloader to load from. Depending on the need, return either:
        - (img_feat, label), concept_feat will be loaded in the model
        - (the dot product between img_feat and concept_feat, label)
        - if allowing grad to image, provide (image, label)
        - if allowing grad to text, compute concept_feat inside the model        
        """
        self.datasets = {
            mode: ImageFeatDataset(self.img_feat[mode], self.label[mode],
                                   self.on_gpu)
            for mode in ['train', 'val', 'test']
        }

    def train_dataloader(self):
        return DataLoader(
            self.datasets['train'],
            batch_size=self.bs,
            shuffle=True,
            num_workers=self.num_workers if not self.on_gpu else 0,
            pin_memory=True if not self.on_gpu else False)

    def val_dataloader(self):
        return DataLoader(
            self.datasets['val'],
            batch_size=self.bs,
            num_workers=self.num_workers if not self.on_gpu else 0,
            pin_memory=True if not self.on_gpu else False)

    def test_dataloader(self):
        return DataLoader(
            self.datasets['test'],
            batch_size=self.bs,
            num_workers=self.num_workers if not self.on_gpu else 0,
            pin_memory=True if not self.on_gpu else False)

    def predict_dataloader(self):
        test_img_paths = self.get_img_n_shot(self.splits['test'], 'all')[0]
        return DataLoader(
            Dataset_with_name(self.datasets['test'], test_img_paths),
            batch_size=self.bs,
            num_workers=self.num_workers if not self.on_gpu else 0,
            pin_memory=True if not self.on_gpu else False)


class DotProductDataModule(DataModule):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        

    def setup(self, stage):
        """
        Set up datasets for dataloader to load from. Depending on the need, return either:
        - (img_feat, label), concept_feat will be loaded in the model
        - (the dot product between img_feat and concept_feat, label)
        - if allowing grad to image, provide (image, label)
        - if allowing grad to text, compute concept_feat inside the model        
        """
        self.datasets = {
            mode: DotProductDataset(
                self.img_feat[mode],
                self.concept_feat[self.select_idx[:self.num_concept]],
                self.label[mode], self.on_gpu)
            for mode in ['train', 'val', 'test']
        }