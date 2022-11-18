_base_ = 'aircraft_base.py'
n_shots = 4
data_root = 'exp/asso_opt/aircraft/aircraft_4shot_fac'
lr = 5e-5
bs = 64

concept_type = "all_submodular"
concept_select_fn = "submodular"
submodular_weights = [1e7, 0.1]