GPU=0 python anagen_test_eval.py bert_base \
    --from_npy outputs/bert_base.eval_out.npy \
    --use_l1 \
    --anagen_model_dir /home/hansonlu/links/data/anagen_models/anagen_anaph_only_b28_lr_default
