import torch
import argparse
from anagen.speaker_model import LiteralSpeakerModel
from anagen.dataset import AnagenDataset
from anagen.evaluate import create_eval_csv, parse_eval_args

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parse_eval_args(parser)

    # IMPORTANT: must set random seed before initializing model
    if args.random_seed >= 0:
        print("setting random seed to %d" % args.random_seed)
        torch.manual_seed(args.random_seed)

    eval_dataset = AnagenDataset(args.eval_jsonlines, args.eval_batch_size, args.max_segment_len)
    model = LiteralSpeakerModel(args)

    create_eval_csv(args, model, eval_dataset)
