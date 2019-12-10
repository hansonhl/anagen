TRAIN_FILE=/home/hansonlu/links/data/pp_ontonotes/anagen.prev10.max80.train.txt
TEST_FILE=/home/hansonlu/links/data/pp_ontonotes/anagen.prev10.max80.dev.txt

python transformers/examples/run_lm_finetuning.py \
    --output_dir=train_output \
    --model_type=gpt2 \
    --model_name_or_path=gpt2 \
    --do_train \
    --train_data_file=$TRAIN_FILE \
    --do_eval \
    --eval_data_file=$TEST_FILE \
    --anagen
