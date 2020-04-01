import torch
import numpy as np
import logging
import tqdm
import time
import sys

from anagen.utils import combine_subtokens, batch_to_device
from anagen.dataset import AnagenDataset, AnagenDocument, AnagenExample, collate, GPT2_EOS_TOKEN_ID
from anagen.speaker_model import RNNSpeakerModel
from transformers import AnagenGPT2LMHeadModel, GPT2Tokenizer
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler


MAX_SPAN_WIDTH=10

class CorefRSAModel:
    def __init__(self, anteced_top_k, logger=None):
        self.anteced_top_k = anteced_top_k
        self.logger=logger

    def _log_debug(self, msg):
        if self.logger is not None:
            self.logger.debug(msg)
        else:
            print(msg)
    def _log_info(self, msg):
        if self.logger is not None:
            self.logger.info(msg)
        else:
            print(msg, *args)

    def top_k_idxs_along_axis1(self, a):
        idxs = np.argpartition(a, -self.anteced_top_k, axis=1)[:,-self.anteced_top_k:]
        return idxs

    def flatten_sentences(self, sentences):
        res = []
        for s in sentences:
            res += s
        return res

    def get_sentence_starts(self, sentence_map):
        starts = [0]
        curr = 0
        for i in range(len(sentence_map)):
            if sentence_map[i] != curr:
                starts.append(i)
                curr = sentence_map[i]
        starts = np.array(starts)
        return starts

    def l1(self, example, top_span_starts, top_span_ends,
           top_antecedents, top_antecedent_scores, alphas=1.0):
        raise NotImplementedError()

def in_same_cluster(clusters, span1_start, span1_end, span2_start, span2_end):
    clusters_with_span1 = [[span1_start, span1_end] in c for c in clusters]
    clusters_with_span2 = [[span2_start, span2_end] in c for c in clusters]
    return any(x and y for x, y in zip(clusters_with_span1, clusters_with_span2))

class RNNSpeakerRSAModel(CorefRSAModel):
    def __init__(self, model_dir, batch_size, max_segment_len, anteced_top_k,
                 max_num_ctxs_in_batch, device, tokenizer=None, logger=None):
        super(RNNSpeakerRSAModel, self).__init__(anteced_top_k, logger)
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2") if tokenizer is None else tokenizer
        self.s0_model = RNNSpeakerModel.from_checkpoint(model_dir)
        self.batch_size = batch_size
        self.max_segment_len = max_segment_len
        self.max_num_ctxs_in_batch = max_num_ctxs_in_batch
        self.device = device

        ckpt = torch.load(model_dir)
        args = ckpt["args"]
        self.use_speaker_info = args.use_speaker_info if hasattr(args, "use_speaker_info") else False

        self.s0_model.to(device)

    def retokenize(self, orig_words):
        word_idx = 0
        subtok_idx = 0
        gpt_toks = []
        gpt_subtok_to_word_map = []
        gpt_word_to_subtok_start_map = []
        gpt_word_to_subtok_end_map = []
        for word in orig_words:
            subtokens = self.tokenizer.tokenize(word)
            gpt_word_to_subtok_start_map.append(subtok_idx)
            for subtoken in subtokens:
                gpt_toks.append(subtoken)
                gpt_subtok_to_word_map.append(word_idx)
                subtok_idx += 1
            gpt_word_to_subtok_end_map.append(subtok_idx-1)
            word_idx += 1
        return gpt_toks, gpt_subtok_to_word_map, gpt_word_to_subtok_start_map, gpt_word_to_subtok_end_map

    def retokenize_bert(self, bert_subtok_to_word_map):
        bert_word_to_subtok_start_map = []
        bert_word_to_subtok_end_map = []
        prev_word_id = -1
        for subtok_id, word_id in enumerate(bert_subtok_to_word_map):
            if word_id != prev_word_id:
                bert_word_to_subtok_start_map.append(subtok_id)
                if prev_word_id != -1:
                    bert_word_to_subtok_end_map.append(bert_subtok_to_word_map[subtok_id-1])
            prev_word_id = word_id
        bert_word_to_subtok_end_map.append(len(bert_subtok_to_word_map)-1)
        return bert_word_to_subtok_start_map, bert_word_to_subtok_end_map

    def bert_to_orig_speakers(self, speakers, subtoken_map):
        assert len(speakers) == len(subtoken_map)
        res = []
        prev_x = -1
        for speaker, x in zip(speakers, subtoken_map):
            if prev_x != x:
                res.append(speaker)
            prev_x = x
        return res

    def orig_to_gpt_speakers(self, speakers, gpt_word_to_subtok_end_map):
        assert len(speakers) == len(gpt_word_to_subtok_end_map)
        res = []
        j = 0
        for i in range(len(speakers)):
            while j <= gpt_word_to_subtok_end_map[i]:
                res.append(speakers[i])
                j += 1
        return res

    def l1(self, example, top_span_starts, top_span_ends, top_antecedents,
           top_antecedent_scores, alphas=1.0, debug=False, debug_out_file=None):
        # accept a sequence of alphas to run grid search
        assert isinstance(alphas, float) or isinstance(alphas, int)  \
            or isinstance(alphas, list) or isinstance(alphas, tuple)

        if debug:
            if debug_out_file:
                debug_f = open("debug_out_file", "w")
            else:
                debug_f = sys.stdout

        # Series of tokenization and data preprocessing similar to AnagenDataset
        # initialize dataset on this document, use dataset object later to get
        # input into s0.
        dataset = AnagenDataset(jsonlines_file=None, batch_size=self.batch_size,
                                max_span_width=MAX_SPAN_WIDTH,
                                max_num_ctxs_in_batch=self.max_num_ctxs_in_batch,
                                max_segment_len=self.max_segment_len,
                                use_speaker_info=self.use_speaker_info,
                                tokenizer=self.tokenizer)

        # set up tokenization transform from bert to gpt
        bert_toks = self.flatten_sentences(example["sentences"])
        bert_subtok_to_word_map = example["subtoken_map"]
        orig_words = combine_subtokens(bert_toks, bert_subtok_to_word_map, is_bert=True)
        gpt_toks, gpt_subtok_to_word_map, gpt_word_to_subtok_start_map, gpt_word_to_subtok_end_map = self.retokenize(orig_words)
        bert_word_to_subtok_start_map, bert_word_to_subtok_end_map = self.retokenize_bert(bert_subtok_to_word_map)
        bert_subtok_to_word_map = np.array(bert_subtok_to_word_map)
        bert_word_to_subtok_start_map = np.array(bert_word_to_subtok_start_map)
        bert_word_to_subtok_end_map = np.array(bert_word_to_subtok_end_map)
        print("len(orig_words)", len(orig_words))
        print("bert_word_to_subtok_start_map.shape", bert_word_to_subtok_start_map.shape)
        print("bert_word_to_subtok_end_map.shape", bert_word_to_subtok_end_map.shape)
        gpt_subtok_to_word_map = np.array(gpt_subtok_to_word_map)
        gpt_word_to_subtok_start_map = np.array(gpt_word_to_subtok_start_map)
        gpt_word_to_subtok_end_map = np.array(gpt_word_to_subtok_end_map)
        # print("bert_subtok_to_word_map.shape", bert_subtok_to_word_map.shape)
        # print("gpt_subtok_to_word_map.shape", gpt_subtok_to_word_map.shape)
        # print("gpt_word_to_subtok_start_map.shape", gpt_word_to_subtok_start_map.shape)
        # print("gpt_word_to_subtok_end_map.shape", gpt_word_to_subtok_end_map.shape)

        # get starting positions of segments
        bert_segment_lens = np.array([0] + [len(s) for s in example["sentences"]][:-1])
        bert_segment_starts = np.cumsum(bert_segment_lens)
        gpt_segment_starts = gpt_word_to_subtok_start_map[bert_subtok_to_word_map[bert_segment_starts]]
        gpt_segment_starts = np.append(gpt_segment_starts, len(gpt_toks))

        # prepare speaker information
        bert_speakers = self.flatten_sentences(example["speakers"])
        orig_speakers = self.bert_to_orig_speakers(bert_speakers, bert_subtok_to_word_map)
        gpt_speakers = self.orig_to_gpt_speakers(orig_speakers, gpt_word_to_subtok_end_map)
        assert len(gpt_speakers) == len(gpt_toks)

        # prepare document and add to dataset
        doc_key = example["doc_key"]
        segments = []
        for i in range(len(gpt_segment_starts)-1):
            ctx_start, ctx_end = gpt_segment_starts[i], gpt_segment_starts[i+1]
            segments.append(list(gpt_toks)[ctx_start:ctx_end])
        document = AnagenDocument(doc_key=doc_key, segments=segments,
                                  segment_starts=gpt_segment_starts,
                                  subtoken_map=gpt_subtok_to_word_map,
                                  speakers=gpt_speakers,
                                  tokenizer=self.tokenizer)
        dataset.documents[doc_key] = document

        # Extract candidates given l0 probabilities: get top k antecedents
        all_anteced_arr_idxs = self.top_k_idxs_along_axis1(top_antecedent_scores)
        # get span indeces for each one, null anteced has span idx -1.
        all_anteced_span_idxs = np.where(all_anteced_arr_idxs != 0,
            np.take_along_axis(top_antecedents, all_anteced_arr_idxs-1, axis=1), -1)

        # transform starts and ends
        gpt_span_starts = gpt_word_to_subtok_start_map[bert_subtok_to_word_map[top_span_starts]]
        gpt_span_ends = gpt_word_to_subtok_end_map[bert_subtok_to_word_map[top_span_ends]]

        # create examples to add to dataset, logic similar to
        # dataset._process_jsonline() and GPTSpeakerRSAModel.get_s0_input()
        valid_map = []
        valid_examples = [] # valid examples per anaphor
        valid_anteced_arr_idxs = []
        examples = []
        for anaphor_span_idx in range(len(gpt_span_starts)):
            anaphor_start = gpt_span_starts[anaphor_span_idx]
            anaphor_end = gpt_span_ends[anaphor_span_idx]

            valid_arr_idxs = []
            valid_exs = []
            for anteced_arr_idx, anteced_span_idx in \
                zip(all_anteced_arr_idxs[anaphor_span_idx],
                    all_anteced_span_idxs[anaphor_span_idx]):
                if anteced_span_idx >= anaphor_span_idx:
                    valid_map.append(False)
                    continue
                ctx_seg_start_idx, ctx_seg_end_idx = \
                    dataset.get_ctx_seg_idxs(gpt_segment_starts, anaphor_start)
                if anteced_span_idx >= 0:
                    anteced_start = gpt_span_starts[anteced_span_idx]
                    anteced_end = gpt_span_ends[anteced_span_idx]

                    if anteced_end >= anaphor_start:
                        valid_map.append(False)
                        continue
                    ex = AnagenExample(doc_key, anteced_start, anteced_end,
                                       anaphor_start, anaphor_end,
                                       ctx_seg_start_idx, ctx_seg_end_idx)
                else:
                    # null antecedent
                    ex = AnagenExample(doc_key, -1, -1,
                                       anaphor_start, anaphor_end,
                                       ctx_seg_start_idx, ctx_seg_end_idx)
                examples.append(ex)
                valid_map.append(True)

                valid_arr_idxs.append(anteced_arr_idx)
                valid_exs.append(ex)

            valid_anteced_arr_idxs.append(valid_arr_idxs)
            valid_examples.append(valid_exs)
        dataset.docs_to_examples[doc_key] = examples
        dataset.num_examples = len(examples)

        # finish up dataset
        dataset._finalize_batches()

        # print("num valid %d / total %d / expected total %d" % (sum(valid_map), len(valid_map),
        #                                                  all_anteced_span_idxs.shape[0] * all_anteced_span_idxs.shape[1]))
        s0_input = dataset, valid_map, all_anteced_span_idxs.shape[0], all_anteced_span_idxs.shape[1]
        s0_scores = self.s0(s0_input)
        if isinstance(alphas, int) or isinstance(alphas, float):
            s0_scores *= alphas
            if debug:
                clusters = example["clusters"]
                print(clusters)
                valid_map = np.array(valid_map).reshape((all_anteced_span_idxs.shape[0], all_anteced_span_idxs.shape[1]))
                # debug to see scores
                for anaphor_span_idx in range(len(gpt_span_starts)):
                    anaphor_start = gpt_span_starts[anaphor_span_idx]
                    anaphor_end = gpt_span_ends[anaphor_span_idx]
                    anaphor_bert_start = bert_word_to_subtok_start_map[gpt_subtok_to_word_map[anaphor_start]]

                    anaphor_str = document.decode(anaphor_start, anaphor_end)

                    # debug_f.write("anteced stats: (start, end) str: s0_score + score_before = score_after")
                    anteced_arr_idxs = valid_anteced_arr_idxs[anaphor_span_idx]
                    scores = s0_scores[anaphor_span_idx][valid_map[anaphor_span_idx]]
                    exs = valid_examples[anaphor_span_idx]
                    anteced_strs = []
                    for i, ex in enumerate(exs):
                        score_before = top_antecedent_scores[anaphor_span_idx][anteced_arr_idxs[i]]
                        score_after = score_before + scores[i]
                        anteced_str = document.decode(ex.anteced_start, ex.anteced_end)
                        anteced_strs.append(anteced_str)
                        # debug_f.write("  anteced (%d, %d) %s: %.2f + %.2f = %.2f" % (
                        #     ex.anteced_start, ex.anteced_end, anteced_str,
                        #     scores[i], score_before, score_after))

                    old_scores = top_antecedent_scores[anaphor_span_idx][anteced_arr_idxs]
                    prev_best_anteced_i = np.argmax(old_scores)
                    new_scores = top_antecedent_scores[anaphor_span_idx][anteced_arr_idxs] + scores
                    new_best_anteced_i = np.argmax(new_scores)
                    if new_best_anteced_i != prev_best_anteced_i:
                        ctx_seg_start_idx_1 = exs[prev_best_anteced_i].ctx_seg_start_idx
                        ctx_start_1 = document.segment_starts[ctx_seg_start_idx_1]
                        debug_f.write("*******************\n")
                        debug_f.write("anaphor (%d, %d) %s\n" % (anaphor_start, anaphor_end, anaphor_str))
                        debug_f.write("  BEST ANTECED CHANGED:\n")
                        debug_f.write("  anteced (start, end) str: s0_score + score_before = score_after\n")
                        debug_f.write("  prev best in ctx: (%d, %d) %s: %.2f + %.2f = %.2f\n" % (
                            exs[prev_best_anteced_i].anteced_start - ctx_start_1,
                            exs[prev_best_anteced_i].anteced_end - ctx_start_1,
                            anteced_strs[prev_best_anteced_i],
                            scores[prev_best_anteced_i],
                            old_scores[prev_best_anteced_i],
                            new_scores[prev_best_anteced_i]
                        ))
                        if anteced_strs[prev_best_anteced_i] != "<null>":
                            pass
                        ctx_seg_start_idx_2 = exs[new_best_anteced_i].ctx_seg_start_idx
                        ctx_start_2 = document.segment_starts[ctx_seg_start_idx_2]
                        debug_f.write("  new best in ctx : (%d, %d)%s: %.2f + %.2f = %.2f\n" % (
                            exs[new_best_anteced_i].anteced_start - ctx_start_2,
                            exs[new_best_anteced_i].anteced_end - ctx_start_2,
                            anteced_strs[new_best_anteced_i],
                            scores[new_best_anteced_i],
                            old_scores[new_best_anteced_i],
                            new_scores[new_best_anteced_i]
                        ))
                        ctx_end_2 = exs[new_best_anteced_i].anaphor_start - 1
                        debug_f.write("  [context] %s\n" % document.decode(ctx_start_2, ctx_end_2))
                if debug_out_file:
                    debug_f.close()

            # modify scores
            l1_scores = np.copy(top_antecedent_scores)
            for anaphor_span_idx in range(len(gpt_span_starts)):
                anteced_idxs = all_anteced_arr_idxs[anaphor_span_idx]
                l1_scores[anaphor_span_idx][anteced_idxs] += s0_scores[anaphor_span_idx]

            return l1_scores
        else:
            all_l1_scores = []
            for alpha in alphas:
                curr_s0_scores = s0_scores * alpha
                l1_scores = np.copy(top_antecedent_scores)
                for anaphor_span_idx in range(len(gpt_span_starts)):
                    anteced_idxs = all_anteced_arr_idxs[anaphor_span_idx]
                    l1_scores[anaphor_span_idx][anteced_idxs] += curr_s0_scores[anaphor_span_idx]
                all_l1_scores.append(l1_scores)
            return all_l1_scores

    def s0(self, s0_input):
        dataset, valid_map, num_anaphors, num_anteceds = s0_input
        sampler = SequentialSampler(dataset)
        dataloader = DataLoader(dataset, sampler=sampler, batch_size=1,
                                collate_fn=collate)
        self.s0_model.eval()

        all_scores = []
        for batch in dataloader:
            with torch.no_grad():
                batch = batch_to_device(batch, self.device)

                res_dict = self.s0_model(batch)
                scores = res_dict["logits"]

                scramble_idxs = batch["scramble_idxs"]
                anaphor_ids = batch["anaphor_ids"]
                anaphor_ids_padding_mask = batch["anaphor_ids_padding_mask"]

                # transform into original order
                _, unscramble_idxs = torch.sort(scramble_idxs)
                scores = scores.index_select(0, unscramble_idxs) # [batch, max_len, vocab]
                anaphor_ids = anaphor_ids.index_select(0, unscramble_idxs) # [batch, max_length]
                anaphor_ids_padding_mask = anaphor_ids_padding_mask.index_select(0, unscramble_idxs)

                batch_size = scores.shape[0]
                end_toks = self.s0_model.end_tok.unsqueeze(0).repeat(batch_size, 1)
                anaphor_ids = torch.cat((anaphor_ids, end_toks), 1)
                ones = torch.ones(batch_size, device=self.device).unsqueeze(1)
                anaphor_ids_padding_mask = torch.cat((ones, anaphor_ids_padding_mask), 1)

                scores = scores.gather(2, anaphor_ids.unsqueeze(2)).squeeze()
                scores = scores.mul(anaphor_ids_padding_mask).sum(dim=1)
                # print("scores before len normalization", scores)
                scores = scores.div(anaphor_ids_padding_mask.sum(dim=1))
                # print("scores after len normalization", scores)
                # for toks in anaphor_ids:
                #     print(s0_input.decode_ids(toks))
                all_scores += scores.tolist()

        s0_scores = np.zeros(len(valid_map))
        s0_scores[valid_map] = all_scores
        return s0_scores.reshape((num_anaphors, num_anteceds))


class GPTSpeakerRSAModel(CorefRSAModel):
    def __init__(self, model_dir, max_segment_len, anteced_top_k, device, logger=None):
        super(GPTSpeakerRSAModel, self).__init__(anteced_top_k, logger)
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_dir)
        model = AnagenGPT2LMHeadModel.from_pretrained(model_dir)
        model.to(device)
        model.eval()
        self.model = model
        self.device = device
        self.max_segment_len = max_segment_len
        self.s0_max_sequence_len = 1024
        self.anteced_top_k = anteced_top_k
        self.logger = logger

    def get_ctx_start(self, sentence_starts, anaphor_start, anteced_start):
        anaphor_sent_idx = np.argmax(sentence_starts > anaphor_start) - 1
        if anteced_start is not None:
            anteced_sent_idx = np.argmax(sentence_starts > anteced_start) - 1

        if anteced_start is None or anaphor_start - sentence_starts[anteced_sent_idx] <= self.max_segment_len:
            # make ctx as long as possible under max segment len
            ctx_start_sent_idx = np.argmax(anaphor_start - sentence_starts <= self.max_segment_len)
        else:
            # anteced is very far from anaphor
            ctx_start_sent_idx = anteced_sent_idx
        return sentence_starts[ctx_start_sent_idx]

    def convert_tokens_to_string(self, tokens, append_tag=None):
        # filter [CLS] and [SEP], merge tokens prefixed by ##.
        tokens = list(filter(lambda x: x != "[CLS]" and x != "[SEP]", tokens))

        res = " ".join(tokens) \
                 .replace(" ##", "") \
                 .replace(" [CLS] ", " ") \
                 .replace(" [SEP] ", " ") \
                 .strip()
        res += append_tag
        return res

    def get_single_l1_score(self, input_str, anaphor_str):
        # based off of anagen/evaluate.py
        input_toks = self.tokenizer.encode(input_str)
        anaphor_toks = self.tokenizer.encode(anaphor_str)
        lh_context = torch.tensor(input_toks, dtype=torch.long, device=self.device)
        generated = lh_context

        with torch.no_grad():
            inputs = {"input_ids": generated}
            outputs = self.model(**inputs)
            debug_f.write(outputs[0].shape)
            next_token_logits = outputs[0][-1, :]

    def l1(self, example, top_span_starts, top_span_ends, top_antecedents, top_antecedent_scores):
        # TODO: apply scores from s0 to top_antecedent_scores
        # get top k antecedents
        all_anteced_arr_idxs = self.top_k_idxs_along_axis1(top_antecedent_scores)
        # get span indeces for each one, null anteced has span idx -1.
        all_anteced_span_idxs = np.where(all_anteced_arr_idxs != 0,
            np.take_along_axis(top_antecedents, all_anteced_arr_idxs-1, axis=1), -1)

        # get bert-style string tokens in one-dim list
        raw_bert_toks = self.flatten_sentences(example["sentences"])
        # get starting positions of sentences
        sentence_starts = self.get_sentence_starts(example["sentence_map"])

        for anaphor_span_idx in range(top_antecedents.shape[0]):
            anaphor_start = top_span_starts[anaphor_span_idx]
            anaphor_end = top_span_ends[anaphor_span_idx]
            anteced_arr_idxs = all_anteced_arr_idxs[anaphor_span_idx]
            anteced_span_idxs = all_anteced_span_idxs[anaphor_span_idx]
            anaphor_toks = raw_bert_toks[anaphor_start:anaphor_end+1]

            # valid mask, may be optimized out of the for loop
            anteced_valid_mask = anteced_span_idxs < anaphor_span_idx
            anteced_valid_arr_idxs = anteced_arr_idxs[anteced_valid_mask]

            s0_input = self.get_s0_input(anaphor_toks, anaphor_span_idx,
                                         anaphor_start, anaphor_end,
                                         anteced_span_idxs,
                                         top_span_starts, top_span_ends,
                                         sentence_starts, raw_bert_toks)

            # feed into GPT model to get probabilities
            scores = self.s0(s0_input) #[batch]
            top_antecedent_scores[anaphor_span_idx][anteced_valid_arr_idxs] += scores

        return top_antecedent_scores

    def get_s0_input(anaphor_toks, anaphor_span_idx,
                     anaphor_start, anaphor_end, anteced_span_idxs,
                     top_span_starts, top_span_ends,
                     sentence_starts, raw_bert_toks):
        anaphor_str = self.convert_tokens_to_string(anaphor_toks, append_tag=" </anaphor>")
        all_input_strs = []
        # the following are for debug:
        all_anteced_valid_span_idxs = []
        all_anteced_starts = []
        all_anteced_strs = []
        for anteced_span_idx in anteced_span_idxs:
            if anteced_span_idx >= anaphor_span_idx:
                anteced_start = int(top_span_starts[anteced_span_idx])
                anteced_end = int(top_span_ends[anteced_span_idx])
                # debug_f.write("  invalid anteced %d: (%d, %d) %s" % (anteced_span_idx, anteced_start, anteced_end, " ".join(raw_bert_toks[anteced_start:anteced_end+1])))
                continue
            elif anteced_span_idx >= 0:
                # assume arr idx is the correct ordering of spans start token, and then by length
                anteced_start = int(top_span_starts[anteced_span_idx])
                anteced_end = int(top_span_ends[anteced_span_idx])
                ctx_start = self.get_ctx_start(sentence_starts, anaphor_start, anteced_start)
                # debug_f.write("  anteced %d: (%d, %d) %s" % (anteced_span_idx, anteced_start, anteced_end, " ".join(raw_bert_toks[anteced_start:anteced_end+1])))
                all_anteced_valid_span_idxs.append(anteced_span_idx)
                all_anteced_starts.append(anteced_start)
                all_anteced_strs.append(" ".join(raw_bert_toks[anteced_start:anteced_end+1]))
            else:
                ctx_start = self.get_ctx_start(sentence_starts, anaphor_start, None)
                all_anteced_valid_span_idxs.append(-1)
                all_anteced_starts.append(-1)
                all_anteced_strs.append("<Null anteced>")
                # debug_f.write("  Null anteced")

            ctx_tokens = raw_bert_toks[ctx_start:anaphor_start]

            if anteced_span_idx >= 0 and anteced_span_idx < anaphor_span_idx \
                and anteced_end < anaphor_start:
                # ignore nested anaphors, treat them as null antecedent
                ctx_tokens.insert(anteced_start - ctx_start, "<anteced>")
                ctx_tokens.insert(anteced_end - ctx_start + 2, "</anteced>")

            input_str = self.convert_tokens_to_string(ctx_tokens, append_tag=" <anaphor>")
            all_input_strs.append(input_str)

        return anaphor_str, all_input_strs

    def prepare_batch(self, context_strs, anaphor_str):
        anaphor_toks = self.tokenizer.encode(anaphor_str)
        # debug_f.write("anaphor_toks %s" % anaphor_toks)
        # debug_f.write("anaphor_toks in str: %s" % self.tokenizer.convert_ids_to_tokens(anaphor_toks))
        anaphor_len = len(anaphor_toks)
        input_toks = [(self.tokenizer.encode(s) + anaphor_toks) for s in context_strs]
        # truncate from left to ensure maximum input sequence length
        input_toks = [torch.tensor(s[-self.s0_max_sequence_len:]) for s in input_toks]
        input_lens = np.array([x.shape[0] for x in input_toks])
        scramble_idxs = np.argsort(input_lens)

        # the following is based off of hansonlu/transformers/run_lm_finetuning
        # my_scaled_collate
        pad_token_id = self.tokenizer.eos_token_id
        sorted_batch = sorted(input_toks, key=lambda b: b.shape[0], reverse=True)
        lengths = torch.LongTensor([x.shape[0] for x in sorted_batch])
        padded = torch.nn.utils.rnn.pad_sequence(sorted_batch, batch_first=True,
            padding_value=pad_token_id) # [batch, max_input_len]
        max_input_len = padded.shape[1]
        padding_mask = (torch.arange(max_input_len)[None] < lengths[:,None]) # [batch, max_input_len]
        anaphor_mask = (torch.arange(max_input_len)[None] >= (lengths-anaphor_len-1)[:,None]) \
            & (torch.arange(max_input_len)[None] < (lengths - 1)[:,None])
        # Note: tensor[None] is equiv to tensor.unsqueeze(0)
        #       tensor[:, None] equiv to tensor.unsqueeze(1)

        return padded, padding_mask, anaphor_mask, anaphor_toks, lengths, scramble_idxs

    def s0(self, s0_input):
        anaphor_str, context_strs = s0_input
        with torch.no_grad():
            batch, attention_mask, anaphor_mask, anaphor_toks, lengths, scramble_idxs \
                = self.prepare_batch(context_strs, anaphor_str)
            # [batch, maxlen], [batch, maxlen], [batch, maxlen], [anaphor_len], [batch], npy[batch],
            anaphor_len = len(anaphor_toks)

            # the following is based off of AnagenGPT2LMHeadModel.forward()
            inputs = batch.to(self.device)
            labels = batch.to(self.device)
            attention_mask = attention_mask.type(torch.FloatTensor).to(self.device)
            anaphor_mask = anaphor_mask.to(self.device)
            anaphor_toks = torch.tensor(anaphor_toks).to(self.device)

            # get model output: unnormalized logit distrib for every token in input
            transformer_outputs = self.model.transformer(inputs, attention_mask=attention_mask)
            hidden_states = transformer_outputs[0] # [batch, max_len, hidden_dim]
            lm_logits = self.model.lm_head(hidden_states)  # [batch, max_len, vocab]

            # filter to get anaphor logit distrib
            flat_logits = lm_logits.view(-1, lm_logits.shape[-1]) # [batch*max_len, vocab]
            flat_anaphor_mask = anaphor_mask.view(-1) # [batch*max_len]
            flat_anaphor_idxs = torch.nonzero(flat_anaphor_mask).squeeze() # [batch*anaphor_len]
            flat_anaphor_logits = flat_logits.index_select(0, flat_anaphor_idxs) # [batch*anaphor_len, vocab]
            flat_anaphor_idxs = anaphor_toks.repeat(batch.shape[0]) # [batch*anaphor_len]

            # get logits of target anaphor tokens
            anaphor_tgt_logits = flat_anaphor_logits.gather(1, flat_anaphor_idxs[:,None]).squeeze() #[batch*anaphor_len,1]=>[batch*anaphor_len]
            anaphor_tgt_logits = anaphor_tgt_logits.view(batch.shape[0], -1) # [batch,anaphor_len]

            # get score by taking mean, convert to numpy, and unscramble
            scrambled_scores = anaphor_tgt_logits.mean(1).cpu().numpy() # [batch]
            unscramble_idxs = np.argsort(scramble_idxs)
            scores = scrambled_scores[unscramble_idxs]
            return scores

            # debug_f.write("lengths", lengths)
            # debug_f.write("flat_anaphor_idxs", flat_anaphor_idxs)
            # debug_f.write("check flat_anaphor_idxs.shape, expected [%d], actual %s" % (self.anteced_top_k * anaphor_len, flat_anaphor_idxs.shape))
            # debug_f.write("check flat_anaphor_logits.shape, expected [%d,vocab], actual %s" % (self.anteced_top_k * anaphor_len, flat_anaphor_logits.shape))



        # old L1 fxn
    # def l1(self, example, top_span_starts, top_span_ends, top_antecedents, top_antecedent_scores):
    #     # TODO: apply scores from s0 to top_antecedent_scores
    #     debug_f.write("********** Running l1 ***********")
    #     # get top k antecedents
    #     all_anteced_arr_idxs = self.top_k_idxs_along_axis1(top_antecedent_scores)
    #     # get span indeces for each one, null anteced has span idx -1.
    #     all_anteced_span_idxs = np.where(all_anteced_arr_idxs != 0,
    #         np.take_along_axis(top_antecedents, all_anteced_arr_idxs-1, axis=1), -1)
    #
    #     # get bert-style string tokens in one-dim list
    #     raw_bert_toks = self.flatten_sentences(example["sentences"])
    #     # get starting positions of sentences
    #     sentence_starts = self.get_sentence_starts(example["sentence_map"])
    #
    #     for anaphor_span_idx in range(top_antecedents.shape[0]):
    #         anaphor_start = top_span_starts[anaphor_span_idx]
    #         anaphor_end = top_span_ends[anaphor_span_idx]
    #         anteced_arr_idxs = all_anteced_arr_idxs[anaphor_span_idx]
    #         anteced_span_idxs = all_anteced_span_idxs[anaphor_span_idx]
    #         anaphor_toks = raw_bert_toks[anaphor_start:anaphor_end+1]
    #         anaphor_str = self.convert_tokens_to_string(anaphor_toks, append_tag=" </anaphor>")
    #
    #         # valid mask, may be optimized out of the for loop
    #         anteced_valid_mask = anteced_span_idxs < anaphor_span_idx
    #         anteced_valid_arr_idxs = anteced_arr_idxs[anteced_valid_mask]
    #
    #         all_input_strs = [] # [batch]
    #
    #         # the following are for debug:
    #         all_anteced_valid_span_idxs = []
    #         all_anteced_starts = []
    #         all_anteced_strs = []
    #         for anteced_span_idx in anteced_span_idxs:
    #             if anteced_span_idx >= anaphor_span_idx:
    #                 anteced_start = int(top_span_starts[anteced_span_idx])
    #                 anteced_end = int(top_span_ends[anteced_span_idx])
    #                 # debug_f.write("  invalid anteced %d: (%d, %d) %s" % (anteced_span_idx, anteced_start, anteced_end, " ".join(raw_bert_toks[anteced_start:anteced_end+1])))
    #                 continue
    #             elif anteced_span_idx >= 0:
    #                 # assume arr idx is the correct ordering of spans start token, and then by length
    #                 anteced_start = int(top_span_starts[anteced_span_idx])
    #                 anteced_end = int(top_span_ends[anteced_span_idx])
    #                 ctx_start = self.get_ctx_start(sentence_starts, anaphor_start, anteced_start)
    #                 # debug_f.write("  anteced %d: (%d, %d) %s" % (anteced_span_idx, anteced_start, anteced_end, " ".join(raw_bert_toks[anteced_start:anteced_end+1])))
    #                 all_anteced_valid_span_idxs.append(anteced_span_idx)
    #                 all_anteced_starts.append(anteced_start)
    #                 all_anteced_strs.append(" ".join(raw_bert_toks[anteced_start:anteced_end+1]))
    #             else:
    #                 ctx_start = self.get_ctx_start(sentence_starts, anaphor_start, None)
    #                 all_anteced_valid_span_idxs.append(-1)
    #                 all_anteced_starts.append(-1)
    #                 all_anteced_strs.append("<Null anteced>")
    #                 # debug_f.write("  Null anteced")
    #
    #             ctx_tokens = raw_bert_toks[ctx_start:anaphor_start]
    #
    #             if anteced_span_idx >= 0 and anteced_span_idx < anaphor_span_idx \
    #                 and anteced_end < anaphor_start:
    #                 # ignore nested anaphors, treat them as null antecedent
    #                 ctx_tokens.insert(anteced_start - ctx_start, "<anteced>")
    #                 ctx_tokens.insert(anteced_end - ctx_start + 2, "</anteced>")
    #
    #             input_str = self.convert_tokens_to_string(ctx_tokens, append_tag=" <anaphor>")
    #             all_input_strs.append(input_str)
    #
    #         # feed into GPT model to get probabilities
    #         scores = self.s0(all_input_strs, anaphor_str) #[batch]
    #         top_antecedent_scores[anaphor_span_idx][anteced_valid_arr_idxs] += scores
    #
            # # debug to see scores
            # debug_f.write("anteced stats: span_idx (start) str: s0_score/score_before/score_after")
            # for i, (input_str, span_idx, start_idx, antecstr) in \
            #     enumerate(zip(all_input_strs, all_anteced_valid_span_idxs, all_anteced_starts, all_anteced_strs)):
            #     score_before = top_antecedent_scores[anaphor_span_idx][anteced_valid_arr_idxs[i]]
            #     score_after = score_before + scores[i]
            #     debug_f.write("  anteced %d (%d) %s: %.2f/%.2f/%.2f" % (
            #         span_idx, start_idx, antecstr,
            #         scores[i], score_before, score_after))
            #
            # old_scores = top_antecedent_scores[anaphor_span_idx][anteced_valid_arr_idxs]
            # prev_best_anteced_i = np.argmax(old_scores)
            # new_scores = top_antecedent_scores[anaphor_span_idx][anteced_valid_arr_idxs] + scores
            # new_best_anteced_i = np.argmax(new_scores)
            # if new_best_anteced_i != prev_best_anteced_i:
            #     debug_f.write("*******************")
            #     debug_f.write("anaphor %d: (%d, %d) %s" % (anaphor_span_idx, anaphor_start, anaphor_end, " ".join(raw_bert_toks[anaphor_start:anaphor_end+1])))
            #     debug_f.write("  BEST ANTECED CHANGED:")
            #     debug_f.write("  stats: span_idx (start) str: s0_score/score_before/score_after")
            #     debug_f.write("  prev_best: %d (%d) %.2f/%.2f/%.2f %s\n[context] %s" % (
            #         all_anteced_valid_span_idxs[prev_best_anteced_i],
            #         all_anteced_starts[prev_best_anteced_i],
            #         old_scores[prev_best_anteced_i],
            #         scores[prev_best_anteced_i],
            #         new_scores[prev_best_anteced_i],
            #         all_anteced_strs[prev_best_anteced_i],
            #         all_input_strs[prev_best_anteced_i]
            #     ))
            #     debug_f.write("  new_best: %d (%d) %.2f/%.2f/%.2f %s\n[context] %s" % (
            #         all_anteced_valid_span_idxs[new_best_anteced_i],
            #         all_anteced_starts[new_best_anteced_i],
            #         old_scores[new_best_anteced_i],
            #         scores[new_best_anteced_i],
            #         new_scores[new_best_anteced_i],
            #         all_anteced_strs[new_best_anteced_i],
            #         all_input_strs[new_best_anteced_i]
            #     ))


        return top_antecedent_scores
