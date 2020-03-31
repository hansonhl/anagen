import torch
import argparse
from anagen.speaker_model import RNNSpeakerModel
from anagen.dataset import AnagenDataset
from anagen.train import evaluate, parse_train_args
from transformers import GPT2Tokenizer


def main():
    parser = argparse.ArgumentParser()
    args = parse_train_args(parser)

    # IMPORTANT: must set random seed before initializing model
    if args.random_seed >= 0:
        print("setting random seed to %d" % args.random_seed)
        torch.manual_seed(args.random_seed)

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    eval_dataset = AnagenDataset(jsonlines_file=args.eval_jsonlines,
                                 batch_size=args.eval_batch_size,
                                 max_span_width=args.max_span_width,
                                 max_num_ctxs_in_batch=args.max_num_ctxs_in_batch,
                                 max_segment_len=args.max_segment_len,
                                 use_speaker_info=args.use_speaker_info,
                                 tokenizer=tokenizer)

    model = RNNSpeakerModel.from_checkpoint(args.model_load_path)

    ckpt = torch.load(args.model_load_path)
    global_step = ckpt["global_step"]
    epoch = ckpt["epoch"]

    print("*** Evaluating model that has been trained for %d epochs" % global_step)

    evaluate(args, model, eval_dataset, global_step)


if __name__ == "__main__":
    main()
