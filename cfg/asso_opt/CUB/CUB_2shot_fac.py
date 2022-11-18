_base_ = 'CUB_base.py'
n_shots = 2
data_root = 'exp/asso_opt/CUB/CUB_2shot_fac'
lr = 5e-5
bs = 64

concept_type = "all_submodular"