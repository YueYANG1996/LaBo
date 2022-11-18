_base_ = 'HAM10000_base.py'
n_shots = 1
data_root = 'exp/asso_opt/HAM10000/HAM10000_1shot_fac'

lr = 1e-3
bs = 4

max_epochs = 20000

concept_type = "all_submodular"
concept_select_fn = "submodular"
submodular_weights = [1e7, 0.1]