_base_ = 'CIFAR100_base.py'
n_shots = 2
data_root = 'exp/asso_opt/CIFAR100/CIFAR100_2shot_fac'
lr = 1e-5
bs = 32

concept_type = "all_submodular"
concept_select_fn = "submodular"
submodular_weights = [1e7, 2.5]