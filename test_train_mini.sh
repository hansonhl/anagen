#!/bin/bash

python run_train.py \
    --train_jsonlines data/dev.english.256.twodoc.anagen.jsonlines \
    --eval_jsonlines data/dev.english.256.twodoc.anagen.jsonlines \
    --gpu \
    --train_batch_size 8 \
    --max_num_ctxs_in_batch 1 \
    --log_steps 5 \
    --train_epochs 5 \
    --model_save_path output/test_model_save \
    --eval_and_save_by_epoch
