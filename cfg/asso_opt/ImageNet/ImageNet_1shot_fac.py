_base_ = 'ImageNet_base.py'
n_shots = 1
data_root = 'exp/asso_opt/ImageNet/ImageNet_1shot_fac'
lr = 1e-5
bs = 128

concept_type = "all_submodular"
concept_select_fn = "submodular"
submodular_weights = [1e8, 0]