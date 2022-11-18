_base_ = 'CIFAR100_base.py'
n_shots = "all"
data_root = 'exp/asso_opt/CIFAR100/CIFAR100_allshot_fac'
lr = 1e-5
bs = 512

concept_type = "all_submodular"
concept_select_fn = "submodular"
submodular_weights = [1e7, 0]