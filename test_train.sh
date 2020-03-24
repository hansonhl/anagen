#!/bin/bash
PP_DIR=/home/hansonlu/links/data/pp_coref_anagen
python anagen_test_train.py \
    --train_jsonlines $PP_DIR/train.english.256.anagen.jsonlines \
    --eval_jsonlines $PP_DIR/dev.english.256.anagen.jsonlines \
    --gpu \
    --train_epochs 3
