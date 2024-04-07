EXP_DIR=./content/exp

python jiant/scripts/download_data/runscript.py \
    download \
    --tasks copa \
    --output_path ${EXP_DIR}/tasks
# python jiant/proj/simple/runscript.py \
#     run \
#     --run_name simple \
#     --exp_dir ${EXP_DIR}/ \
#     --data_dir ${EXP_DIR}/tasks \
#     --hf_pretrained_model_name_or_path roberta-base \
#     --tasks mrpc \
#     --train_batch_size 16 \
#     --num_train_epochs 3