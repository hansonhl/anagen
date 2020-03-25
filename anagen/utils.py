import numpy as np
# from transformers import GPT2LMHeadModel, GPT2Tokenizer

import json
import torch
import argparse

def prune_antec_scores(top_antecedents, top_antecedent_scores, top_k):
    idxs = np.argpartition(top_antecedent_scores, top_k, axis=1)[:,-top_k:]
    pruned_top_antecedent_scores = np.take_along_axis(top_antecedent_scores,
                                                      idxs, axis=1)
    idxs = idxs - 1
    pruned_top_antecedents = np.take_along_axis(top_antecedents, idxs, axis=1)

    return pruned_top_antecedents, pruned_top_antecedent_scores

# copied from util.flatten()
def flatten(l):
  return [item for sublist in l for item in sublist]

def batch_to_device(batch, device):
    for k, v in batch.items():
        if torch.is_tensor(v):
            batch[k] = v.to(device)
    return batch
def parse_eval_args(parser):
    return parser.parse_args()

""" check weights to confirm backprop went through """
def check_state_dict(model, optimizer=None):
    print("****** checking state dict of model ******")
    # print(state_dict.keys())
    speaker_state_dict = model.state_dict()
    gpt2_state_dict = model.gpt2_model.state_dict()
        # print("optimizer.state_dict()['param_groups']", optimizer.state_dict()['param_groups'])

    print("gpt2_model.wte.requires_grad", model.gpt2_model.wte.weight.requires_grad)
    print("null_anteced_emb.requires_grad", model.null_anteced_emb.requires_grad)
    print("gpt2_model.wte.weight.grad", model.gpt2_model.wte.weight.grad)
    return (gpt2_state_dict["h.0.attn.bias"][0][0][0][:10].tolist(),
            gpt2_state_dict["wte.weight"][9][:10].tolist(),
            speaker_state_dict["token_embedding.weight"][9][:10].tolist(),
            model.null_anteced_emb.data[:10].tolist(),
            model.hidden_to_logits.weight[0][:10].tolist(),
            )
