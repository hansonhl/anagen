# from transformers import GPT2LMHeadModel, GPT2Tokenizer
import json
import torch
import numpy as np
import argparse
import random

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
            model.hidden_to_logits.weight[0][:10].tolist())

def combine_subtokens(toks, subtoken_map, is_bert=False):
    res = []
    prev_x = -1
    curr_word = ""
    for tok, x in zip(toks, subtoken_map):
        if is_bert and (tok == "[CLS]" or tok == "[SEP]"):
            continue
        if prev_x != x and prev_x != -1:
            res.append(curr_word)
            curr_word = ""
        if is_bert:
            tok = tok.strip("#")
        curr_word += tok
        prev_x = x
    if curr_word != "":
        res.append(curr_word)

    return res

def invert_subtoken_map(subtok_to_word, bert_toks=None):
    word_to_subtok_start = []
    word_to_subtok_end = []
    prev_word_id = -1
    for subtok_id, word_id in enumerate(subtok_to_word):
        if bert_toks:
            if bert_toks[subtok_id] == "[CLS]" or bert_toks[subtok_id] == "[SEP]":
                continue
        if word_id != prev_word_id:
            word_to_subtok_start.append(subtok_id)
            if prev_word_id != -1:
                word_to_subtok_end.append(subtok_id-1)
        prev_word_id = word_id
    if bert_toks:
        word_to_subtok_end.append(len(subtok_to_word)-2)
    else:
        word_to_subtok_end.append(len(subtok_to_word)-1)
    return word_to_subtok_start, word_to_subtok_end

def set_random_seed(seed):
    print("setting random seed to %d" % seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
