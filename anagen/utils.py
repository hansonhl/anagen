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
