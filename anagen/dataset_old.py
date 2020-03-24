# Code copied from
# https://stackoverflow.com/questions/57277214/multi-gpu-training-of-allennlp-coreference-resolution

import torch
from allennlp.common import Params
from allennlp.data import Vocabulary
from allennlp.data.dataset_readers import ConllCorefReader
from allennlp.data.dataset_readers.dataset_utils import Ontonotes
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenCharactersIndexer


class AnagenDoc:
    def __init__(self, text_field, pairs, max_prev_anteceds, max_span_width):
        self.tok_str_list = [t.text for t in text_field]
        self.max_prev_anteceds = max_prev_anteceds
        self.pairs = pairs

def get_anagen_docs_from_file(
        path: str,
        max_prev_anteceds: int,
        max_span_width: int = 10):
    """
    Use ConllCorefReader in the allennlp package to read in labeled
    coreference data from OntoNotes 5.0 preprocessed using the CoNLL 2012
    shared task format

    args:
        path: path to dataset ending in `_gold_conll` generated with
            allennlp/scripts/compile_coref_data.sh
            Requires OntoNotes 5.0 dataset.
        maximum number of closest antecedents to pair up with an anaphor
    """
    dataset_reader = ConllCorefReader(max_span_width, {"tokens": SingleIdTokenIndexer(),
                                           "token_characters": TokenCharactersIndexer()})
    data = dataset_reader.read(path)
    res_docs = []
    print("hello")

    for doc_i, inst in enumerate(data):
        spans = inst["spans"]
        span_labels = inst["span_labels"]
        print(inst["metadata"].metadata)
        return
        num_spans = len(spans)
        pairs = []
        cluster_dict = {}
        for span, cluster_id in zip(spans, span_labels):
            if cluster_id != -1:
                tup_curr_span = (span.span_start, span.span_end)
                if cluster_id not in cluster_dict:
                    cluster_dict[cluster_id] = [tup_curr_span]
                else:
                    num_anteceds = len(cluster_dict[cluster_id])
                    for anteced_i in range(min(max_prev_anteceds, num_anteceds)):
                        anteced_span = cluster_dict[cluster_id][-anteced_i]
                        pairs.append((anteced_span, tup_curr_span))
                    cluster_dict[cluster_id].append(tup_curr_span)

        res_docs.append(AnagenDoc(inst["text"], pairs, max_prev_anteceds, max_span_width))

    return res_docs


def generate_anagen_dataset_from_file(
        path: str,
        out_path: str,
        max_prev_anteceds: int = 3,
        max_span_width: int = 10,
        max_context_length: int = 80,
        context_mode: str = "ALL_COMPLETE_SENTS"
    ):
    """
    Use ConllCorefReader in the allennlp package to read in labeled
    coreference data from OntoNotes 5.0 preprocessed using the CoNLL 2012
    shared task format

    args:
        path: path to dataset ending in `_gold_conll` generated with
            allennlp/scripts/compile_coref_data.sh
            Requires OntoNotes 5.0 dataset.
        max_prev_anteceds:
            maximum number of closest antecedents to pair up with an anaphor
        max_span_width:
            maximum span width, used for ConllCorefReader
        max_context_length:
            maximum context length
    """
    dataset_reader = ConllCorefReader(max_span_width, {"tokens": SingleIdTokenIndexer(),
                                           "token_characters": TokenCharactersIndexer()})
    data = dataset_reader.read(path)

    out_f = open(out_path, "w")

    for doc_i, inst in enumerate(data):
        spans = inst["spans"]
        span_labels = inst["span_labels"]
        orig_text = inst["text"]
        cluster_dict = {}
        for span, cluster_id in zip(spans, span_labels):
            if cluster_id != -1:
                curr_span = (span.span_start, span.span_end)
                curr_start, curr_end = curr_span
                if cluster_id not in cluster_dict:
                    cluster_dict[cluster_id] = [curr_span]
                else:
                    # span is not the first in a cluster, iterate through antecedents
                    num_anteceds = len(cluster_dict[cluster_id])
                    for anteced_i in range(min(max_prev_anteceds, num_anteceds)):
                        anteced_start, anteced_end = cluster_dict[cluster_id][-anteced_i]
                        # some cases where antecedent contains anaphor, ignore
                        if anteced_end >= curr_end:
                            continue

                        # prevent long contexts
                        if curr_start - anteced_start > max_context_length:
                            break

                        # get context according to context mode
                        if context_mode == "ALL_COMPLETE_SENTS" \
                            or context_mode == "NEAREST_COMPLETE_SENT":
                            # find the longest context within the limit that starts with a full stop
                            idx = anteced_start - 1
                            prev_full_stop_idx = anteced_start
                            while idx >= 0 and curr_start - idx <= max_context_length:
                                if orig_text[idx].text == ".":
                                    prev_full_stop_idx = min(idx, prev_full_stop_idx)
                                idx -= 1
                            if prev_full_stop_idx != anteced_start:
                                ctx_start = prev_full_stop_idx + 1
                                ctx_end = curr_start - 1 # inclusive
                            else:
                                # do not look for any more antecedents
                                break
                        else:
                            raise NotImplementedError

                        # context must contain antecedent
                        if not (ctx_start <= anteced_start and anteced_end <= ctx_end):
                            print(" ".join([t.text for t in orig_text[:curr_end+1]]))
                            print("ctx_start = %d, anteced_start = %d, anteced_end = %d, ctx_end = %d" \
                                  % (ctx_start, anteced_start, anteced_end, ctx_end))
                            print("curr_start = %d, curr_end = %d" % (curr_start, curr_end))
                            return

                        # write raw text
                        tok_str_list = [orig_text[i].text for i in
                                        range(ctx_start, anteced_start)]
                        tok_str_list += ["<anteced>"]
                        tok_str_list += [orig_text[i].text for i in
                                         range(anteced_start, anteced_end + 1)]
                        tok_str_list += ["</anteced>"]
                        tok_str_list += [orig_text[i].text for i in
                                         range(anteced_end + 1, curr_start)]
                        tok_str_list += ["<anaphor>"]
                        tok_str_list += [orig_text[i].text for i in
                                         range(curr_start, curr_end + 1)]
                        tok_str_list += ["</anaphor>"]
                        tok_str_list += ["<|endoftext|>"]
                        out_str = " ".join(tok_str_list)
                        out_f.write(out_str + "\n")

                    cluster_dict[cluster_id].append(curr_span)
    out_f.close()
