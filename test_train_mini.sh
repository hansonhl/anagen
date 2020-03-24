#!/bin/bash

python anagen_test_train.py \
    --train_jsonlines data/dev.english.256.twodoc.anagen.jsonlines \
    --gpu \
    --train_batch_size 16 \
    --max_num_ctxs_in_batch 8 \
    --log_steps 5 \
    --train_epochs 20
