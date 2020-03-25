#!/bin/bash

python run_train.py \
    --train_jsonlines data/dev.english.256.twodoc.anagen.jsonlines \
    --eval_jsonlines data/dev.english.256.twodoc.anagen.jsonlines \
    --gpu \
    --model_load_path /home/hansonlu/links/data/coref_anagen_models/coref_anagen_0324_basic_batch_768_best.ckpt \
    --unfreeze_gpt2 \
    --train_batch_size 16 \
    --max_num_ctxs_in_batch 1 \
    --log_steps 5 \
    --train_epochs 20 \
    --eval_and_save_steps 10 \
    --model_save_path output/test_model_save
