 python3 hw5_final.py --model model --action train --cell GRU --train_path $1  --semi_path $2
 python3 hw5_final.py --model model_semi --action semi --load_model model --cell GRU --train_path $1 --semi_path $2
