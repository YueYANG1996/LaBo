# $1: mode (asso_opt/clip_finetune), 
# $2: number of shots
# $3: dataset (flower/food101)

python main.py --cfg cfg/$1/$3/$3_$2shot_fac.py --work-dir exp/$1/$3/$3_$2shot_fac --func $1_main ${@:4}