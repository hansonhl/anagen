import torch
import argparse
from anagen.speaker_model import RNNSpeakerModel
from anagen.dataset import AnagenDataset
from anagen.train import evaluate, parse_train_args
from transformers import GPT2Tokenizer
from anagen.utils import set_random_seed

def main():
    parser = argparse.ArgumentParser()
    args = parse_train_args(parser)

    # IMPORTANT: must set random seed before initializing model
    if args.random_seed >= 0:
        set_random_seed(args.random_seed)

    device = torch.device("cuda" if args.gpu else "cpu")

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    eval_dataset = AnagenDataset(input_file=args.input_file,
                                 data_augment=args.data_augment,
                                 data_augment_input_file=args.eval_data_augment_file,
                                 batch_size=args.eval_batch_size,
                                 max_span_width=args.max_span_width,
                                 max_num_ctxs_in_batch=args.max_num_ctxs_in_batch,
                                 max_segment_len=args.max_segment_len,
                                 use_speaker_info=args.use_speaker_info,
                                 tokenizer=tokenizer)

    model = RNNSpeakerModel.from_checkpoint(args.model_load_path)
    model.to(device)

    ckpt = torch.load(args.model_load_path)
    global_step = ckpt["global_step"]
    epoch = ckpt["epoch"]

    print("*** Evaluating model that has been trained for %d epochs" % epoch)

    evaluate(args, model, eval_dataset, global_step)


if __name__ == "__main__":
    main()
