import os
import torch
import argparse
import logging
import numpy as np
import time

from anagen.rsa_model import GPTSpeakerRSAModel, RNNSpeakerRSAModel

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("name")
    parser.add_argument("--from_npy", type=str)
    parser.add_argument("--use_l1", action="store_true")
    parser.add_argument("--s0_model_type", type=str)
    parser.add_argument("--s0_model_path", type=str)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_segment_len", type=int, default=512)
    parser.add_argument("--max_num_ctxs_in_batch", type=int, default=8)
    parser.add_argument("--anteced_top_k", type=int, default=5)

    args = parser.parse_args()

    device = torch.device("cuda" if "GPU" in os.environ and torch.cuda.is_available() else "cpu")

    logger = logging.getLogger("anagen_eval")
    logger.propogate = False
    # while logger.handlers:
        # logger.handlers.pop()
    logger.setLevel(logging.DEBUG)
    # hdlr = logging.StreamHandler()
    # hdlr.setLevel(logging.DEBUG)
    # fmtr = logging.Formatter("[%(asctime)s %(name)s %(levelname)s] %(message)s",
    #                          datefmt\="%H:%M:%S")
    # hdlr.setFormatter(fmtr)
    # logger.addHandler(hdlr)

    if args.s0_model_type in ["gpt", "GPT"]:
        rsa_model = GPTSpeakerRSAModel(args.s0_model_path, device,
                                  max_segment_len=args.max_segment_len,
                                  anteced_top_k=args.anteced_top_k,
                                  logger=logger)
    elif args.s0_model_type in ["rnn", "RNN"]:
        rsa_model = RNNSpeakerRSAModel(args.s0_model_path, args.batch_size,
                                       args.max_segment_len,
                                       args.anteced_top_k,
                                       args.max_num_ctxs_in_batch,
                                       device,
                                       logger=logger)
    # setup logger
    with open(args.from_npy, "rb") as f:
        from_npy_dict = np.load(f)
        data_dicts = from_npy_dict.item().get("data_dicts")

    total_time = 0
    total_docs = 0
    for example_num, data_dict in enumerate(data_dicts):
        example = data_dict["example"]
        tensorized_example = data_dict["tensorized_example"]
        loss = data_dict["loss"]
        top_span_starts = data_dict["top_span_starts"]
        top_span_ends = data_dict["top_span_ends"]
        top_antecedents = data_dict["top_antecedents"]
        top_antecedent_scores = data_dict["top_antecedent_scores"]

        logger.info("running l1 for sentence %d" % example_num)
        start_time = time.time()
        rsa_model.l1(example, top_span_starts, top_span_ends, top_antecedents, top_antecedent_scores)
        duration = time.time() - start_time
        logger.info("sentence %d finished, took %.2f s" % (example_num, duration))
        total_time += duration
        total_docs += 1

        break
    print("average time per sentence: %.2f s" % (total_time / total_docs))
