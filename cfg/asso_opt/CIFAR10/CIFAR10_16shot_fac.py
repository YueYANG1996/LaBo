_base_ = 'CIFAR10_base.py'
n_shots = 16
data_root = 'exp/asso_opt/CIFAR10/CIFAR10_16shot_fac'
lr = 1e-4
bs = 32

concept_type = "all_submodular"
concept_select_fn = "submodular"
submodular_weights = [1e7, 10]