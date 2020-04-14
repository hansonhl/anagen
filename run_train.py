import torch
import argparse
from anagen.speaker_model import RNNSpeakerModel
from anagen.dataset import AnagenDataset
from anagen.train import train, parse_train_args
from anagen.utils import set_random_seed
from transformers import GPT2Tokenizer

def main():
    parser = argparse.ArgumentParser()
    args = parse_train_args(parser)

    # IMPORTANT: must set random seed before initializing model
    if args.random_seed >= 0:
        set_random_seed(args.random_seed)

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    train_dataset = AnagenDataset(input_file=args.train_input_file,
                                  data_augment=args.data_augment,
                                  data_augment_file=args.train_data_augment_file,
                                  batch_size=args.train_batch_size,
                                  max_span_width=args.max_span_width,
                                  max_num_ctxs_in_batch=args.max_num_ctxs_in_batch,
                                  max_segment_len=args.max_segment_len,
                                  use_speaker_info=args.use_speaker_info,
                                  shuffle=args.shuffle_examples,
                                  tokenizer=tokenizer)

    if args.eval_input_file:
        eval_dataset = AnagenDataset(input_file=args.eval_input_file,
                                     data_augment=args.data_augment,
                                     data_augment_file=args.eval_data_augment_file,
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
