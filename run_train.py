import torch
import argparse
from anagen.speaker_model import RNNSpeakerModel
from anagen.dataset import AnagenDataset
from anagen.train import train, parse_train_args
from transformers import GPT2Tokenizer

def main():
    parser = argparse.ArgumentParser()
    args = parse_train_args(parser)

    # IMPORTANT: must set random seed before initializing model
    if args.random_seed >= 0:
        print("setting random seed to %d" % args.random_seed)
        torch.manual_seed(args.random_seed)

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    train_dataset = AnagenDataset(jsonlines_file=args.train_jsonlines,
                                  batch_size=args.train_batch_size,
                                  max_span_width=args.max_span_width,
                                  max_num_ctxs_in_batch=args.max_num_ctxs_in_batch,
                                  max_segment_len=args.max_segment_len,
                                  use_speaker_info=args.use_speaker_info,
                                  tokenizer=tokenizer)

    if args.eval_jsonlines:
        eval_dataset = AnagenDataset(jsonlines_file=args.eval_jsonlines,
                                     batch_size=args.eval_batch_size,
                                     max_span_width=args.max_span_width,
                                     max_num_ctxs_in_batch=args.max_num_ctxs_in_batch,
                                     max_segment_len=args.max_segment_len,
                                     use_speaker_info=args.use_speaker_info,
                                     tokenizer=tokenizer)
    else:
        eval_dataset = None
    model = RNNSpeakerModel(args)

    train(args, model, train_dataset, eval_dataset)

if __name__ == "__main__":
    main()
