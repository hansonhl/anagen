""" Usage: python anagen_test_dataset.py """

from anagen.dataset import *
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler
from transformers import GPT2Tokenizer

example_in_path = "data/dev.english.256.onedoc.anagen.jsonlines"
# use tokenizer from pretrained model already downloaded to my machine
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

dataset = AnagenDataset(example_in_path, batch_size=8,
                        max_span_width=10,
                        max_num_ctxs_in_batch=2,
                        tokenizer=tokenizer)


sampler = SequentialSampler(dataset)
dataloader = DataLoader(dataset, sampler=sampler,
                              batch_size=1,
                              collate_fn=collate)

for i, batch in enumerate(dataloader):
    print("*** checking batch %d *** " % i)
    # print("batch[ctx_ids].shape", batch["ctx_ids"].shape)
    # print("batch[ctx_ids_padding_mask].shape", batch["ctx_ids_padding_mask"].shape)
    # print("batch[ctx_lens]", batch["ctx_lens"])
    # print("batch[anteced_starts].shape", batch["anteced_starts"].shape)
    # print("batch[anteced_ends].shape", batch["anteced_ends"].shape)
    # print("batch[anaphor_ids].shape", batch["anaphor_ids"].shape)
    # print("batch[anaphor_ids_padding_mask].shape", batch["anaphor_ids_padding_mask"].shape)
    # print("batch[speaker_info].shape", batch["speaker_info"].shape)
    doc = dataset.documents[batch["doc_key"]]
    # r = [0, 1, 2] if batch["anteced_starts"].shape[0] >= 3 else [0]
    r = list(range(batch["anteced_starts"].shape[0]))
    for i in r:
        ctx_i = batch["ctx_set_idxs"][i]
        ctx = batch["ctx_ids"][ctx_i]
        ctx_start = batch["ctx_starts"][ctx_i]
        print("ctx_start", ctx_start)
        anteced_start_in_ctx = batch["anteced_starts"][i]
        anteced_end_in_ctx = batch["anteced_ends"][i]
        anteced_start = (ctx_start + anteced_start_in_ctx).item() if anteced_start_in_ctx >= 0 else -1
        anteced_end = (ctx_start + anteced_end_in_ctx).item() if anteced_end_in_ctx >= 0 else -1

        anaphor_start = (ctx_start + batch["anaphor_starts"][i]).item()
        anaphor_end = (anaphor_start + batch["anaphor_lens"][i] - 1).item()
        anaphor_ids = batch["anaphor_ids"][i]

        anteced_str = doc.decode(anteced_start, anteced_end)
        anaphor_str = doc.decode(anaphor_start, anaphor_end)
        anaphor_str_in_batch = dataset.decode_ids(anaphor_ids)

        print("antecedent (%d, %d) anaphor (%d, %d)" % (anteced_start, anteced_end, anaphor_start, anaphor_end))
        print("[anteced]", anteced_str, "[anaphor]", anaphor_str)
        print("[anaphor_str_in_batch]", anaphor_str_in_batch)

        anteced_raw_speaker = doc.raw_speakers[anteced_start] if anteced_start >= 0 else "<null>"
        anaphor_raw_speaker = doc.raw_speakers[anaphor_start]
        speaker_info_in_batch = batch["speaker_info"][i].item()

        # anteced_str = dataset.get_span_toks(ctx, anteced_start, anteced_end)
        # anaphor_str = dataset.decode(anaphor_ids.tolist())

        print("[anteced_speaker]", anteced_raw_speaker, "[anaphor_speaker]", anaphor_raw_speaker, "[in batch]", speaker_info_in_batch)
        if (anteced_raw_speaker == anaphor_raw_speaker) ^ (speaker_info_in_batch == 1):
            "@@@@@@@@@@@@ WRONG SPEAKER INFO @@@@@@@@@@@@@@@@@@@"
