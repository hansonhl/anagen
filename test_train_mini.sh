#!/bin/bash

python run_train.py \
    --train_jsonlines data/dev.english.256.twodoc.anagen.jsonlines \
    --eval_jsonlines data/dev.english.256.twodoc.anagen.jsonlines \
    --gpu \
    --unfreeze_gpt2 \
    --train_batch_size 16 \
    --max_num_ctxs_in_batch 8 \
    --log_steps 5 \
    --train_epochs 3 \
    --eval_and_save_steps 100
