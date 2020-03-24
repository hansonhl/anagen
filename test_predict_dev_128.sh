#!/bin/bash
GPU=0
# input_file=/home/hansonlu/links/data/coref_data/dev.english.128.jsonlines
input_file=/home/hansonlu/anagen/coref/cased_config_vocab/dev.english.head1.jsonlines
# input_file=/home/hansonlu/anagen/coref/cased_config_vocab/trial.jsonlines
output_file=/home/hansonlu/anagen/coref/outputs/dev.english.head1.out.jsonlines
npy_file=/home/hansonlu/anagen/coref/outputs/dev.english.head1.out.npy
python predict.py bert_base $input_file $output_file $npy_file
