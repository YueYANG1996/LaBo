_base_ = 'ImageNet_base.py'
n_shots = "all"
data_root = 'exp/asso_opt/ImageNet/ImageNet_allshot_fac'
lr = 1e-5
bs = 2048
max_epochs = 1000

concept_type = "all_submodular"
concept_select_fn = "submodular"
submodular_weights = [1e8, 0]