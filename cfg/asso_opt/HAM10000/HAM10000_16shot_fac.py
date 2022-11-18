_base_ = 'HAM10000_base.py'
n_shots = 16
data_root = 'exp/asso_opt/HAM10000/HAM10000_16shot_fac'
init_val = 0.1

lr = 1e-3
bs = 16

concept_type = "all_submodular"
concept_select_fn = "submodular"
submodular_weights = [1e7, 15]