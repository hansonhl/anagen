import os
import torch
import argparse
import logging
from anagen.model import CorefRSAModel
from anagen.analysis import test_eval

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("name")
    parser.add_argument("--from_npy", type=str)
    parser.add_argument("--use_l1", action="store_true")
    parser.add_argument("--anagen_model_dir", type=str)
    parser.add_argument("--max_segment_len", type=int, default=128)
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
    #                          datefmt="%H:%M:%S")
    # hdlr.setFormatter(fmtr)
    # logger.addHandler(hdlr)

    rsa_model = CorefRSAModel(args.anagen_model_dir, device,
                              max_segment_len=args.max_segment_len,
                              anteced_top_k=args.anteced_top_k,
                              logger=logger)
    # setup logger

    test_eval(rsa_model, args, logger=logger)
