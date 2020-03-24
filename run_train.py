import torch
import argparse
from anagen.speaker_model import LiteralSpeakerModel
from anagen.dataset import AnagenDataset
from anagen.train import train, parse_train_args

def main():
    parser = argparse.ArgumentParser()
    args = parse_train_args(parser)

    # IMPORTANT: must set random seed before initializing model
    if args.random_seed >= 0:
        print("setting random seed to %d" % args.random_seed)
        torch.manual_seed(args.random_seed)

    train_dataset = AnagenDataset(args.train_jsonlines,
                                  args.train_batch_size,
                                  args.max_num_ctxs_in_batch,
                                  args.max_segment_len)

    if args.eval_jsonlines:
        eval_dataset = AnagenDataset(args.eval_jsonlines,
                                     args.eval_batch_size,
                                     args.max_num_ctxs_in_batch,
                                     args.max_segment_len)
    else:
        eval_dataset = None
    model = LiteralSpeakerModel(args)

    train(args, model, train_dataset, eval_dataset)

if __name__ == "__main__":
    main()
