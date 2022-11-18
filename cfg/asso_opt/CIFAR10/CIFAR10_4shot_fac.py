_base_ = 'CIFAR10_base.py'
n_shots = 4
data_root = 'exp/asso_opt/CIFAR10/CIFAR10_4shot_fac'
lr = 1e-4
bs = 8

concept_type = "all_submodular"
concept_select_fn = "submodular"
submodular_weights = [1e7, 5]