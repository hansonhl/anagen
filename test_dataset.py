""" Usage: python anagen_test_dataset.py """

from anagen.dataset import *
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler
from transformers import GPT2Tokenizer

example_in_path = "data/dev.english.256.onedoc.anagen.jsonlines"
# use tokenizer from pretrained model already downloaded to my machine
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

dataset = AnagenDataset(example_in_path, batch_size=8)

sampler = SequentialSampler(dataset)
dataloader = DataLoader(dataset, sampler=sampler,
                              batch_size=1,
                              collate_fn=collate)

for i, batch in enumerate(dataloader):
    print("*** checking batch %d *** " % i)
    print("batch[ctx_ids].shape", batch["ctx_ids"].shape)
    print("batch[ctx_ids_padding_mask].shape", batch["ctx_ids_padding_mask"].shape)
    print("batch[ctx_lens]", batch["ctx_lens"])
    print("batch[anteced_starts].shape", batch["anteced_starts"].shape)
    print("batch[anteced_ends].shape", batch["anteced_ends"].shape)
    print("batch[anaphor_ids].shape", batch["anaphor_ids"].shape)
    print("batch[anaphor_ids_padding_mask].shape", batch["anaphor_ids_padding_mask"].shape)

    # r = [0, 1, 2] if batch["anteced_starts"].shape[0] >= 3 else [0]
    r = list(range(batch["anteced_starts"].shape[0]))
    print("#### range len %d" % len(r))
    for i in r:
        ctx_i = batch["ctx_set_idxs"][i]
        ctx = batch["ctx_ids"][ctx_i]
        anaphor_ids = batch["anaphor_ids"][i]
        anteced_start = batch["anteced_starts"][i]
        anteced_end = batch["anteced_ends"][i]
        anteced_str = dataset.get_span_toks(ctx, anteced_start, anteced_end)
        anaphor_str = dataset.decode(anaphor_ids.tolist())
        print("[anteced]", anteced_str, "[anaphor]", anaphor_str)
