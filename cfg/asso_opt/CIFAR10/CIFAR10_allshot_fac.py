_base_ = 'CIFAR10_base.py'
n_shots = "all"
data_root = 'exp/asso_opt/CIFAR10/CIFAR10_allshot_fac'
lr = 1e-4
bs = 512

concept_type = "all_submodular"
concept_select_fn = "submodular"
submodular_weights = [1e7, 5]