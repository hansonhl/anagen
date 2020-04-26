GPU=0 python test_eval_rsa.py \
    --from_npy output/bert_base.eval_out.npy \
    --use_l1 \
    --s0_model_type gpt \
    --s0_model_path /home/hansonlu/links/data/rnn_anagen_models/rnn_anagen_0331_fintuned_spk_dist_best.ckpt


# --s0_model_path /home/hansonlu/links/data/anagen_models/anagen_anaph_only_b28_lr_default
