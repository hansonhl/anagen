#!/bin/bash

python run_train.py \
    --train_input_file data/dev.english.256.twodoc.anagen.jsonlines \
    --eval_input_file data/dev.english.256.twodoc.anagen.jsonlines \
    --gpu \
    --train_batch_size 8 \
    --max_num_ctxs_in_batch 1 \
    --log_steps 5 \
    --train_epochs 2 \
    --sum_start_end_emb \
    --use_speaker_info \
    --use_distance_info \
    --shuffle_examples \
    --data_augment "null_from_l0" \
    --train_data_augment_file data/bert_base.dev_head2.npy \
    --eval_data_augment_file data/bert_base.dev_head2.npy \
    --data_augment_max_span_width 10 \
    --model_save_path output/test_model_save \
    --eval_and_save_by_epoch 0.25 \
    --save_latest_state 
