# Code copied from
# https://stackoverflow.com/questions/57277214/multi-gpu-training-of-allennlp-coreference-resolution

import torch

# the following is old test code

# from allennlp.common import Params
# from allennlp.data import Vocabulary
# from allennlp.data.dataset_readers import ConllCorefReader
# from allennlp.data.dataset_readers.dataset_utils import Ontonotes
# from allennlp.data.iterators import PassThroughIterator
# from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenCharactersIndexer
#
# dataset_reader = ConllCorefReader(10, {"tokens": SingleIdTokenIndexer(),
#                                        "token_characters": TokenCharactersIndexer()})
# test_val_path = "/home/hansonlu/links/data/pp_ontonotes/dev.english.10part.v4_gold_conll"
# test_val_data = dataset_reader.read(test_val_path)
#
# # iterator = PassThroughIterator()
#
# max_prev_anteceds = 3
# use_allen_field = False
# res = []
# text_fields = []
#
# for doc_i, inst in enumerate(test_val_data):
#     spans = inst["spans"]
#     span_labels = inst["span_labels"]
#     text_fields.append(inst["text"])
#     num_spans = len(spans)
#     num_mentions = 0
#     doc_res = []
#     cluster_dict = {}
#     for span, cluster_id in zip(spans, span_labels):
#         if cluster_id != -1:
#             num_mentions += 1
#             if cluster_id not in cluster_dict:
#                 cluster_dict[cluster_id] = [span]
#             else:
#                 num_anteceds = len(cluster_dict[cluster_id])
#                 for anteced_i in range(min(max_prev_anteceds, num_anteceds)):
#                     anteced_span = cluster_dict[cluster_id][-anteced_i]
#                     if doc_i == 0:
#                         print(anteced_span)
#                     if not use_allen_field:
#                         tup_anteced_span = (anteced_span.span_start, anteced_span.span_end)
#                         tup_span = (span.span_start, span.span_end)
#                         doc_res.append((tup_anteced_span, tup_span))
#                     else:
#                         doc_res.append((anteced_span, span))
#                 cluster_dict[cluster_id].append(span)
#
#     res.append(doc_res)
#
# # print(type(text_fields[0][0]))
#
# for doc_i, doc in enumerate(res):
#     print("@@@@@@@@@@@@@@@@@ doc %d, len %d" % (doc_i, len(text_fields[doc_i])))
#     for i, pair in enumerate(doc):
#         anteced_span, anaphor_span = pair[0], pair[1]
#         anteced_str = " ".join([t.text for t in text_fields[doc_i][anteced_span[0]:anteced_span[1]+1]])
#         anaphor_str = " ".join([t.text for t in text_fields[doc_i][anaphor_span[0]:anaphor_span[1]+1]])
#         print(anteced_str + "; " + anaphor_str)

# from anagen.dataset import get_anagen_docs_from_file
# import pickle
#
# test_val_path = "/home/hansonlu/links/data/pp_ontonotes/dev.english.v4_gold_conll"
# out_path = "/home/hansonlu/links/data/pp_ontonotes/dev.pickle"
#
# docs = get_anagen_docs_from_file(test_val_path, max_prev_anteceds=3)
#
# with open(out_path, "wb") as f:
#     pickle.dump(docs, f)
#
# pickle_path = "/home/hansonlu/links/data/pp_ontonotes/dev.10part.pickle"
#
# with open(pickle_path, "rb") as f:
#     docs = pickle.load(f)
#
# first_d = docs[0]
#
# print(" ".join(first_d.tok_str_list))
# for anteced_span, anaphor_span in first_d.pairs:
#     anteced_str = " ".join(first_d.tok_str_list[anteced_span[0]:anteced_span[1]+1])
#     anaphor_str = " ".join(first_d.tok_str_list[anaphor_span[0]:anaphor_span[1]+1])
#     print(anteced_str + " " + str(anteced_span) + "; " + anaphor_str + " " + str(anaphor_span))

from tagged_dataset import generate_anagen_dataset_from_file
example_in_path = "/home/hansonlu/links/data/pp_ontonotes/dev.english.10part.v4_gold_conll"
example_out_path = "/home/hansonlu/links/data/pp_ontonotes/dev.10part.txt"

dev_in_path = "/home/hansonlu/links/data/pp_ontonotes/dev.english.v4_gold_conll"
dev_out_path = "/home/hansonlu/links/data/pp_ontonotes/anagen.prev10.max80.dev.txt"

train_in_path = "/home/hansonlu/links/data/pp_ontonotes/train.english.v4_gold_conll"
train_out_path = "/home/hansonlu/links/data/pp_ontonotes/anagen.prev10.max80.train.txt"

generate_anagen_dataset_from_file(
        path=dev_in_path,
        out_path=dev_out_path,
        max_prev_anteceds=10
)

generate_anagen_dataset_from_file(
        path=train_in_path,
        out_path=train_out_path,
        max_prev_anteceds=10
)
