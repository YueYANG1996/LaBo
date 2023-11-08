# $1: number of shots
# $2: dataset (flower/food101)

python main.py --cfg cfg/asso_opt/$1/$1_$2shot_fac.py --work-dir exp/asso_opt/$2/$2_$1shot_fac --func $asso_opt_main ${@:3}
