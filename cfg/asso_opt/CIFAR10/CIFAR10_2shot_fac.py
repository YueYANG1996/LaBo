_base_ = 'CIFAR10_base.py'
n_shots = 2
data_root = 'exp/asso_opt/CIFAR10/CIFAR10_2shot_fac'
lr = 5e-4
bs = 4

concept_type = "all_submodular"
concept_select_fn = "submodular"
submodular_weights = [1e7, 5]