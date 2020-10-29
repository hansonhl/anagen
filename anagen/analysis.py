import numpy as np
import time

def count_ratio():
    with open("outputs/dev.english.128.out.npy", "rb") as f:
        res = np.load(f)

    total_mentions = 0
    null_antec_mentions = 0

    for example_dict in res:
        antec_scores = example_dict["top_antecedent_scores"];
        max_score = antec_scores.max(1)
        non_zero = np.count_nonzero(max_score)
        total_mentions += max_score.shape[0]
        null_antec_mentions += max_score.shape[0] - non_zero

    print("null antec mentions:", null_antec_mentions)
    print("total mentions:", total_mentions)
    print("ratio: ", null_antec_mentions / total_mentions)
