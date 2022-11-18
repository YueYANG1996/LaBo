_base_ = 'ImageNet_base.py'
n_shots = 16
data_root = 'exp/asso_opt/ImageNet/ImageNet_16shot_fac'
lr = 1e-5
bs = 1024

concept_type = "all_submodular"
concept_select_fn = "submodular"
submodular_weights = [1e8, 0]