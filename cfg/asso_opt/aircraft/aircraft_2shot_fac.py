_base_ = 'aircraft_base.py'
n_shots = 2
data_root = 'exp/asso_opt/aircraft/aircraft_2shot_fac'
lr = 5e-5
bs = 32

concept_type = "all_submodular"
concept_select_fn = "submodular"
submodular_weights = [1e7, 1]