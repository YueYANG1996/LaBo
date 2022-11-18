_base_ = 'RESISC45_base.py'
n_shots = 2
data_root = 'exp/asso_opt/RESISC45/RESISC45_2shot_fac'
lr = 5e-5
bs = 16

concept_type = "all_submodular"
concept_select_fn = "submodular"
submodular_weights = [1e7, 5]