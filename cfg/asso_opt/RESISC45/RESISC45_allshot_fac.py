_base_ = 'RESISC45_base.py'
n_shots = "all"
data_root = 'exp/asso_opt/RESISC45/RESISC45_allshot_fac'
lr = 5e-5
bs = 256

concept_type = "all_submodular"
concept_select_fn = "submodular"
submodular_weights = [1e7, 15]