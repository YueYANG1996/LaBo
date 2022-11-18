_base_ = 'HAM10000_base.py'
n_shots = 8
data_root = 'exp/asso_opt/HAM10000/HAM10000_8shot_fac'
init_val = 0.01

lr = 1e-3
bs = 8

concept_type = "all_submodular"
concept_select_fn = "submodular"
submodular_weights = [1e7, 10]