_base_ = 'flower_base.py'
n_shots = 8
data_root = 'exp/asso_opt/flower/flower_8shot_fac'
lr = 1e-5
bs = 128

concept_type = "all_submodular"
concept_select_fn = "submodular"
submodular_weights = [1e7, 10]