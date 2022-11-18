_base_ = 'CUB_base.py'
n_shots = 8
data_root = 'exp/asso_opt/CUB/CUB_8shot_fac'
lr = 5e-5
bs = 256

concept_type = "all_submodular"
concept_select_fn = "submodular"
submodular_weights = [1e7, 0]