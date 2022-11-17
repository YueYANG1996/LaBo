import torch as th
import random
from tqdm import tqdm
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


def get_tSNE_embed(x):
    tSNE = TSNE(n_components=2, init='random')
    return tSNE.fit_transform(x)


def clip_score(img_feat, concept_feat, n_shots, num_images_per_class):
    num_cls = len(num_images_per_class)
    scores_mean = th.empty((concept_feat.shape[0], num_cls))
    start_loc = 0
    for i in range(num_cls):
        end_loc = sum(num_images_per_class[:i+1])
        scores_mean[:, i] = (concept_feat @ img_feat[start_loc:end_loc].t()).mean(dim=-1)
        start_loc = end_loc
    return scores_mean


def mi_score(img_feat, concept_feat, n_shots, num_images_per_class):
    num_cls = len(num_images_per_class)
    scores_mean = clip_score(img_feat, concept_feat, n_shots, num_images_per_class) # Sim(c,y)
    normalized_scores = scores_mean / (scores_mean.sum(dim=0) * num_cls) # Sim_bar(c,y)
    margin_x = normalized_scores.sum(dim=1) # sum_y in Y Sim_bar(c,y)
    margin_x = margin_x.reshape(-1, 1).repeat(1, num_cls)
    # compute MI and PMI
    pmi = th.log(normalized_scores / (margin_x * 1 / num_cls)) # log Sim_bar(c,y) / sum_y in Y Sim_bar(c,y) / N = log(Sim_bar(c|y))
    mi = normalized_scores * pmi  # Sim_bar(c,y)* log(Sim_bar(c|y))
    mi = mi.sum(dim=1)
    return mi, scores_mean


def mi_select(img_feat, concept_feat, n_shots, num_images_per_class, *args):
    mi, _ = mi_score(img_feat, concept_feat, n_shots, num_images_per_class)
    _, selected_idx = th.sort(mi, descending=True)
    return selected_idx


def clip_score_select(img_feat, concept_feat, n_shots, num_images_per_class, *args):
    scores_mean = clip_score(img_feat, concept_feat, n_shots, num_images_per_class)
    best_scores_over_cls = scores_mean.max(dim=-1)[0]
    _, selected_idx = th.sort(best_scores_over_cls, descending=True)
    return selected_idx


def group_clip_select(img_feat, concept_feat, n_shots, concept2cls, num_concepts, num_images_per_class, *args):
    assert num_concepts > 0
    num_cls = len(num_images_per_class)
    scores = clip_score(img_feat, concept_feat, n_shots, num_images_per_class).max(dim=-1)[0]

    selected_idx = []
    concept2cls = th.from_numpy(concept2cls).long()
    num_concepts_per_cls = num_concepts // num_cls
    for i in tqdm(range(num_cls)):
        cls_idx = th.where(concept2cls == i)[0]
        _, idx_for_cls_idx = th.topk(scores[cls_idx], num_concepts_per_cls)
        global_idx = cls_idx[idx_for_cls_idx]
        selected_idx.extend(global_idx)
    return th.tensor(selected_idx)


def group_mi_select(img_feat, concept_feat, n_shots, concept2cls, num_concepts, num_images_per_class, submodular_weights, *args):
    assert num_concepts > 0
    num_cls = len(num_images_per_class)
    scores, _ = mi_score(img_feat, concept_feat, n_shots, num_images_per_class)
    take_all = False
    selected_idx = []
    num_concepts_per_cls = int(np.ceil(num_concepts / num_cls))
    concept2cls = th.from_numpy(concept2cls).long()
    for i in tqdm(range(num_cls)):
        cls_idx = th.where(concept2cls == i)[0]
        if len(cls_idx) == 0: continue

        elif len(cls_idx) < num_concepts_per_cls or (take_all and num_cls < 10):
            global_idx = cls_idx

        else:
            _, idx_for_cls_idx = th.topk(scores[cls_idx], num_concepts_per_cls)
            global_idx = cls_idx[idx_for_cls_idx]

        selected_idx.extend(global_idx)
    return th.tensor(selected_idx)


def clip_score_select_within_cls(img_feat, concept_feat, n_shots, concept2cls):
    # taking from cls2concept and then select top concept within each class
    num_cls = len(img_feat) // n_shots
    scores = concept_feat @ img_feat.t()
    scores = scores.view(concept_feat.shape[0], num_cls, n_shots)
    scores_mean = scores.mean(dim=-1) # (num_concept, num_cls)
    init_cls_id = list(concept2cls.values()) # (num_concept, 1)
    init_cls_id = th.tensor(init_cls_id).view(-1, 1)
    init_score = th.gather(scores_mean, 1, init_cls_id)
    _, selected_idx = th.sort(init_score, descending=True)
    return selected_idx


def compute_class_similarity(img_feat, n_shots):
    # img_feat: n_shots * num_cls x d
    # img_sim: n_shots * num_cls x n_shots * num_cls
    num_cls = len(img_feat) // n_shots
    img_sim = img_feat @ img_feat.T
    class_sim = th.empty((num_cls, num_cls), dtype=th.long)
    for i, row_split in enumerate(th.split(img_sim, n_shots, dim=0)):
        for j, col_split in enumerate(th.split(row_split, n_shots, dim=1)):
            class_sim[i, j] = th.mean(col_split)
    return class_sim / class_sim.max(dim=0).values


def plot(features, selected_idx, filename):
    tsne_features = get_tSNE_embed(features)
    x_selected = tsne_features[selected_idx,0]
    y_selected = tsne_features[selected_idx,1]
    x = tsne_features[:,0]
    y = tsne_features[:,1]
    plt.clf()
    plt.scatter(x, y, s = 1, c ="blue")
    plt.scatter(x_selected, y_selected, s = 10, c ="red", alpha=1)
    plt.savefig('{}.png'.format(filename))


def submodular_select(img_feat, concept_feat, n_shots, concept2cls, num_concepts, num_images_per_class, submodular_weights, *args):
    from apricot import CustomSelection, MixtureSelection, FacilityLocationSelection
    assert num_concepts > 0
    num_cls = len(num_images_per_class)
    
    all_mi_scores, _ = mi_score(img_feat, concept_feat, n_shots, num_images_per_class)
    selected_idx = []
    num_concepts_per_cls = int(np.ceil(num_concepts / num_cls))

    def mi_based_function(X):
        return X[:, 0].sum()
    
    mi_selector = CustomSelection(num_concepts_per_cls, mi_based_function)
    distance_selector = FacilityLocationSelection(num_concepts_per_cls, metric='cosine')

    mi_score_scale = submodular_weights[0]
    facility_weight = submodular_weights[-1]
    
    if mi_score_scale == 0:
        submodular_weights = [0, facility_weight]
    else:
        submodular_weights = [1, facility_weight]

    concept2cls = th.from_numpy(concept2cls).long()
    for i in tqdm(range(num_cls)):
        cls_idx = th.where(concept2cls == i)[0]

        if len(cls_idx) <= num_concepts_per_cls:
            selected_idx.extend(cls_idx)
        else:
            mi_scores = all_mi_scores[cls_idx] * mi_score_scale

            current_concept_features = concept_feat[cls_idx]
            augmented_concept_features = th.hstack([th.unsqueeze(mi_scores, 1), current_concept_features]).numpy()
            selector = MixtureSelection(num_concepts_per_cls, functions=[mi_selector, distance_selector], weights=submodular_weights, optimizer='naive', verbose=False)
            
            selected = selector.fit(augmented_concept_features).ranking
            selected_idx.extend(cls_idx[selected])

    return th.tensor(selected_idx)


def random_select(img_feat, concept_feat, n_shots, concept2cls, num_concepts, num_images_per_class, submodular_weights, *args):
    assert num_concepts > 0
    num_cls = len(num_images_per_class)
    take_all = False
    selected_idx = []
    num_concepts_per_cls = int(np.ceil(num_concepts / num_cls))
    concept2cls = th.from_numpy(concept2cls).long()
    for i in tqdm(range(num_cls)):
        cls_idx = th.where(concept2cls == i)[0]
        if len(cls_idx) == 0: continue

        elif len(cls_idx) < num_concepts_per_cls or (take_all and num_cls < 10):
            global_idx = cls_idx

        else:
            global_idx = random.sample(cls_idx.tolist(), num_concepts_per_cls)

        selected_idx.extend(global_idx)

    return th.tensor(selected_idx)