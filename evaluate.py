import argparse
import logging
from tqdm import tqdm
import sys
from subprocess import check_output

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd

from transformers import GPT2LMHeadModel, GPT2Tokenizer

def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (batch size x vocabulary size)
            top_k > 0: keep only top k tokens with highest probability (top-k filtering).
            top_p > 0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317

        Copied and modified from transformers/examples/run_generation.py
    """
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(dim=1, index=sorted_indices, src=sorted_indices_to_remove)
        logits[indices_to_remove] = filter_value
    return logits

def sample_sequence(model, length, context, num_samples=1, temperature=1,
                    top_k=0, top_p=0.0, repetition_penalty=1.0,
                    device='cpu'):
    """Copied and modified from transformers/examples/run_generation.py"""
    context = torch.tensor(context, dtype=torch.long, device=device)
    context = context.unsqueeze(0).repeat(num_samples, 1)
    generated = context
    with torch.no_grad():
        for _ in range(length):
            inputs = {'input_ids': generated}
            outputs = model(**inputs)  # Note: we could also use 'past' with GPT-2/Transfo-XL/XLNet/CTRL (cached hidden-states)
            next_token_logits = outputs[0][:, -1, :] / (temperature if temperature > 0 else 1.)

            # repetition penalty from CTRL (https://arxiv.org/abs/1909.05858)
            for i in range(num_samples):
                for _ in set(generated[i].tolist()):
                    next_token_logits[i, _] /= repetition_penalty

            filtered_logits = top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)
            if temperature == 0: # greedy sampling:
                next_token = torch.argmax(filtered_logits, dim=-1).unsqueeze(-1)
            else:
                next_token = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1)
            generated = torch.cat((generated, next_token), dim=1)
    return generated

def main():
    """ Copied and modified from transformers/examples/run_generation.py"""
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir", type=str,
                        help="Path to directory that stores the fine-tuned model.")
    parser.add_argument("--eval_path", "-i", type=str,
                        help="pre-processed input data for anagen")
    parser.add_argument("--out_path", "-o", type=str,
                        help="Where to store predictions of the model")
    parser.add_argument("--csv", action="store_true",
                        help="output in csv format")
    parser.add_argument("--readable", action="store_true",
                        help="Produce readable output for easy comparison")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")

    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--max_span_width", type=int, default=10)

    parser.add_argument("--temperature", type=float, default=1.0,
                        help="temperature of 0 implies greedy sampling")
    parser.add_argument("--repetition_penalty", type=float, default=1.0,
                        help="primarily useful for CTRL model; in that case, use 1.2")
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--top_p", type=float, default=0.9)
    args = parser.parse_args()

    assert not (args.csv and args.readable)
    if args.csv:
        assert args.out_path is not None

    args.device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    tokenizer = GPT2Tokenizer.from_pretrained(args.model_dir)
    model = GPT2LMHeadModel.from_pretrained(args.model_dir)
    model.to(args.device)
    model.eval()

    num_lines = int(check_output(['wc', '-l', args.eval_path]).split()[0])
    eval_f = open(args.eval_path, "r")
    out_f = open(args.out_path, "w") if args.out_path else sys.stdout

    if args.csv:
        rows = []

    for line in tqdm(eval_f, total=num_lines):
        line = line.strip()
        x_raw, y_raw = line.split(" <anaphor> ")
        x_raw += " <anaphor>"
        x_tokens = tokenizer.encode(x_raw, add_special_tokens=False)
        out = sample_sequence(
            model=model,
            context=x_tokens,
            num_samples=args.num_samples,
            length=args.max_span_width+2,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            device=args.device
        )
        out = out[:, len(x_tokens):].tolist()
        pred_strs = []
        for o in out:
            text = tokenizer.decode(o, clean_up_tokenization_spaces=True)
            text = text[: text.find("</anaphor>")].strip()
            if "</anteced>" in text:
                text = text[:text.find("</anteced>")].strip()
            pred_strs.append(text)
        ctx_str = x_raw[: x_raw.find("<anaphor>")].strip()
        gold_str = y_raw[: y_raw.find("</anaphor>")].strip()
        if args.csv:
            row = {"context": ctx_str, "gold": gold_str}
            for i, s in enumerate(pred_strs):
                row["pred%d" % i] = s
            rows.append(row)
        elif not args.readable:
            out_f.write(",".join(pred_strs)+"\n")
            pred_str = ", ".join(pred_strs)
            displ_str = "[CTXT] %s\n[GOLD] %s\n[PRED] %s\n\n" % (ctx_str, gold_str, pred_str)
            out_f.write(displ_str)

    eval_f.close()
    if args.out_path:
        out_f.close()
    if args.csv:
        df = pd.DataFrame(rows)
        df.to_csv(args.out_path, index=False)

if __name__ == "__main__":
    main()
