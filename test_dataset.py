""" Usage: python anagen_test_dataset.py """

from anagen.dataset import *
from anagen.utils import set_random_seed, flatten
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler
from transformers import GPT2Tokenizer

def main():
    # example_in_path = "data/dev.english.256.onedoc.anagen.jsonlines"
    example_in_path = "/home/hansonlu/links/data/pp_coref_anagen/train.english.256.anagen.jsonlines"
    data_augment_file = "/home/hansonlu/anagen/coref/outputs/bert_base.train_out.npy"
    # use tokenizer from pretrained model already downloaded to my machine
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    set_random_seed(39393)

    dataset = AnagenDataset(input_file=example_in_path,
                            data_augment="null_from_l0",
                            data_augment_file=data_augment_file,
                            data_augment_max_span_width=12,
                            batch_size=768,
                            max_segment_len=256,
                            max_span_width=10,
                            max_num_ctxs_in_batch=8,
                            use_speaker_info=True,
                            shuffle=True,
                            tokenizer=tokenizer)

    print("finished constructing dataset")

    sampler = SequentialSampler(dataset)
    dataloader = DataLoader(dataset, sampler=sampler,
                                  batch_size=1,
                                  collate_fn=collate)

    anaphor_id_dim0_sum = 0
    anaphor_id_dim1_sum = 0
    ctx_id_dim0_sum = 0
    ctx_id_dim1_sum = 0
    num_batches = 0

    for j, batch in enumerate(dataloader):
        anaphor_id_dim0_sum += batch["anaphor_ids"].shape[0]
        anaphor_id_dim1_sum += batch["anaphor_ids"].shape[1]
        ctx_id_dim0_sum += batch["ctx_ids"].shape[0]
        ctx_id_dim1_sum += batch["ctx_ids"].shape[1]
        num_batches += 1
        # if j % 50 == 0:
        #     print("*** checking batch %d *** " % j)
        #
        #     print("batch[anaphor_ids] [%d, %d] avg [%.2f, %.2f]" %
        #           (batch["anaphor_ids"].shape[0], batch["anaphor_ids"].shape[1],
        #            anaphor_id_dim0_sum / (j + 1), anaphor_id_dim1_sum / (j+1)))
        #     print("batch[ctx_ids] [%d, %d] avg [%.2f, %.2f]" %
        #           (batch["ctx_ids"].shape[0], batch["ctx_ids"].shape[1],
        #            ctx_id_dim0_sum / (j + 1), ctx_id_dim1_sum / (j+1)))
        # print("batch[ctx_ids].shape", batch["ctx_ids"].shape)
        # print("batch[ctx_ids_padding_mask].shape", batch["ctx_ids_padding_mask"].shape)
        # print("batch[ctx_lens]", batch["ctx_lens"])
        # print("batch[anteced_starts].shape", batch["anteced_starts"].shape)
        # print("batch[anteced_ends].shape", batch["anteced_ends"].shape)
        # print("batch[anaphor_ids].shape", batch["anaphor_ids"].shape)
        # print("batch[anaphor_ids_padding_mask].shape", batch["anaphor_ids_padding_mask"].shape)
        # print("batch[speaker_info].shape", batch["speaker_info"].shape)
        # doc = dataset.documents[batch["doc_key"]]
        # r = [0, 1, 2] if batch["anteced_starts"].shape[0] >= 3 else [0]
        # r = list(range(batch["anteced_starts"].shape[0]))
        # for i in r:
        #     # print("******************************************")
        #     ctx_i = batch["ctx_set_idxs"][i]
        #     ctx = batch["ctx_ids"][ctx_i]
        #     ctx_start = batch["ctx_starts"][i]
        #     # print("ctx_start", ctx_start)
        #     anteced_start_in_ctx = batch["anteced_starts"][i]
        #     anteced_end_in_ctx = batch["anteced_ends"][i]
        #     anteced_start = (ctx_start + anteced_start_in_ctx).item() if anteced_start_in_ctx >= 0 else -1
        #     anteced_end = (ctx_start + anteced_end_in_ctx).item() if anteced_end_in_ctx >= 0 else -1
        #
        #     print("anaphor start, end in batch: (%d, %d)" % (batch["anaphor_starts"][i],batch["anaphor_starts"][i] + batch["anaphor_lens"][i]))
        #     anaphor_start = (ctx_start + batch["anaphor_starts"][i]).item()
        #     anaphor_end = (anaphor_start + batch["anaphor_lens"][i] - 1).item()
        #     anaphor_ids = batch["anaphor_ids"][i]
        #
        #     anteced_str = doc.decode(anteced_start, anteced_end)
        #     anaphor_str = doc.decode(anaphor_start, anaphor_end)
        #     anaphor_str_in_batch = dataset.decode_ids(anaphor_ids)
        #
        #     print("antecedent (%d, %d) anaphor (%d, %d)" % (anteced_start, anteced_end, anaphor_start, anaphor_end))
        #     print("[anteced]", anteced_str, "[anaphor]", anaphor_str)
        #     print("[anaphor_str_in_batch]", anaphor_str_in_batch)
        #
        #     # print("len raw toks", len(flatten(doc.segment_toks)))
        #     # print("len_raw_speaker", len(doc.raw_speakers))
        #     anteced_raw_speaker = doc.raw_speakers[anteced_start] if anteced_start >= 0 else "<null>"
        #     anaphor_raw_speaker = doc.raw_speakers[anaphor_start]
        #     speaker_info_in_batch = batch["speaker_info"][i].item()
        #
        #     # anteced_str = dataset.get_span_toks(ctx, anteced_start, anteced_end)
        #     # anaphor_str = dataset.decode(anaphor_ids.tolist())
        #
        #     # print("[anteced_speaker]", anteced_raw_speaker, "[anaphor_speaker]", anaphor_raw_speaker, "[in batch]", speaker_info_in_batch)
        #     if (anteced_raw_speaker == anaphor_raw_speaker) ^ (speaker_info_in_batch == 1):
        #         "@@@@@@@@@@@@ WRONG SPEAKER INFO @@@@@@@@@@@@@@@@@@@"
        #
        #     if i == 10:
        #         break
        # if j == 3:
        #     return

    print("batch[anaphor_ids] avg shape [%.2f, %.2f]" %
          (anaphor_id_dim0_sum / num_batches, anaphor_id_dim1_sum / num_batches))
    print("batch[ctx_ids] avg shape  [%.2f, %.2f]" %
          (ctx_id_dim0_sum / num_batches, ctx_id_dim1_sum / num_batches))

if __name__ == "__main__":
    main()
