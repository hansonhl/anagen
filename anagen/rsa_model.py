import torch
import numpy as np
import logging
import tqdm
from anagen.dataset import GPT2_EOS_TOKEN_ID
from transformers import AnagenGPT2LMHeadModel, GPT2Tokenizer

class CorefRSAModel:
    def __init__(self, model_dir, device, max_segment_len, anteced_top_k, logger=None):
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

    def get_sentence_starts(self, sentence_map):
        starts = [0]
        curr = 0
        for i in range(len(sentence_map)):
            if sentence_map[i] != curr:
                starts.append(i)
                curr = sentence_map[i]
        starts = np.array(starts)
        return starts

    def flatten_sentences(self, sentences):
        res = []
        for s in sentences:
            res += s
        return res

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

    def convert_tokens_to_string(self, tokens, is_ctx=True):
        # filter [CLS] and [SEP], merge tokens prefixed by ##.
        tokens = list(filter(lambda x: x != "[CLS]" and x != "[SEP]", tokens))

        res = " ".join(tokens) \
                 .replace(" ##", "") \
                 .replace(" [CLS] ", " ") \
                 .replace(" [SEP] ", " ") \
                 .strip()
        res += " <anaphor>" if is_ctx else " </anteced>"
        return res

    def top_k_idxs_along_axis1(self, a):
        idxs = np.argpartition(a, -self.anteced_top_k, axis=1)[:,-self.anteced_top_k:]
        return idxs

    def get_single_l1_score(self, input_str, anaphor_str):
        # based off of anagen/evaluate.py
        input_toks = self.tokenizer.encode(input_str)
        anaphor_toks = self.tokenizer.encode(anaphor_str)
        lh_context = torch.tensor(input_toks, dtype=torch.long, device=self.device)
        generated = lh_context

        with torch.no_grad():
            inputs = {"input_ids": generated}
            outputs = self.model(**inputs)
            self._log_debug(outputs[0].shape)
            next_token_logits = outputs[0][-1, :]

    def prepare_batch(self, context_strs, anaphor_str):
        anaphor_toks = self.tokenizer.encode(anaphor_str)
        # self._log_debug("anaphor_toks %s" % anaphor_toks)
        # self._log_debug("anaphor_toks in str: %s" % self.tokenizer.convert_ids_to_tokens(anaphor_toks))
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

    def get_s0_scores(self, context_strs, anaphor_str):
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

            # self._log_debug("lengths", lengths)
            # self._log_debug("flat_anaphor_idxs", flat_anaphor_idxs)
            # self._log_debug("check flat_anaphor_idxs.shape, expected [%d], actual %s" % (self.anteced_top_k * anaphor_len, flat_anaphor_idxs.shape))
            # self._log_debug("check flat_anaphor_logits.shape, expected [%d,vocab], actual %s" % (self.anteced_top_k * anaphor_len, flat_anaphor_logits.shape))

    def l1(self, example, top_span_starts, top_span_ends, top_antecedents, top_antecedent_scores):
        # TODO: apply scores from s0 to top_antecedent_scores
        self._log_debug("********** Running l1 ***********")
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
            anaphor_str = self.convert_tokens_to_string(anaphor_toks, is_ctx=False)

            # valid mask, may be optimized out of the for loop
            anteced_valid_mask = anteced_span_idxs < anaphor_span_idx
            anteced_valid_arr_idxs = anteced_arr_idxs[anteced_valid_mask]

            all_input_strs = [] # [batch]

            # the following are for debug:
            all_anteced_valid_span_idxs = []
            all_anteced_starts = []
            all_anteced_strs = []
            for anteced_span_idx in anteced_span_idxs:
                if anteced_span_idx >= anaphor_span_idx:
                    anteced_start = int(top_span_starts[anteced_span_idx])
                    anteced_end = int(top_span_ends[anteced_span_idx])
                    # self._log_debug("  invalid anteced %d: (%d, %d) %s" % (anteced_span_idx, anteced_start, anteced_end, " ".join(raw_bert_toks[anteced_start:anteced_end+1])))
                    continue
                elif anteced_span_idx >= 0:
                    # assume arr idx is the correct ordering of spans start token, and then by length
                    anteced_start = int(top_span_starts[anteced_span_idx])
                    anteced_end = int(top_span_ends[anteced_span_idx])
                    ctx_start = self.get_ctx_start(sentence_starts, anaphor_start, anteced_start)
                    # self._log_debug("  anteced %d: (%d, %d) %s" % (anteced_span_idx, anteced_start, anteced_end, " ".join(raw_bert_toks[anteced_start:anteced_end+1])))
                    all_anteced_valid_span_idxs.append(anteced_span_idx)
                    all_anteced_starts.append(anteced_start)
                    all_anteced_strs.append(" ".join(raw_bert_toks[anteced_start:anteced_end+1]))
                else:
                    ctx_start = self.get_ctx_start(sentence_starts, anaphor_start, None)
                    all_anteced_valid_span_idxs.append(-1)
                    all_anteced_starts.append(-1)
                    all_anteced_strs.append("<Null anteced>")
                    # self._log_debug("  Null anteced")

                ctx_tokens = raw_bert_toks[ctx_start:anaphor_start]

                if anteced_span_idx >= 0 and anteced_span_idx < anaphor_span_idx \
                    and anteced_end < anaphor_start:
                    # ignore nested anaphors, treat them as null antecedent
                    ctx_tokens.insert(anteced_start - ctx_start, "<anteced>")
                    ctx_tokens.insert(anteced_end - ctx_start + 2, "</anteced>")

                input_str = self.convert_tokens_to_string(ctx_tokens)
                all_input_strs.append(input_str)

            # feed into GPT model to get probabilities
            scores = self.get_s0_scores(all_input_strs, anaphor_str) #[batch]
            top_antecedent_scores[anaphor_span_idx][anteced_valid_arr_idxs] += scores

            # debug to see scores
            # self._log_debug("anteced stats: span_idx (start) str: s0_score/score_before/score_after")
            # for i, (input_str, span_idx, start_idx, antecstr) in \
            #     enumerate(zip(all_input_strs, all_anteced_valid_span_idxs, all_anteced_starts, all_anteced_strs)):
            #     score_before = top_antecedent_scores[anaphor_span_idx][anteced_valid_arr_idxs[i]]
            #     score_after = score_before + scores[i]
            #     self._log_debug("  anteced %d (%d) %s: %.2f/%.2f/%.2f" % (
            #         span_idx, start_idx, antecstr,
            #         scores[i], score_before, score_after))

            # old_scores = top_antecedent_scores[anaphor_span_idx][anteced_valid_arr_idxs]
            # prev_best_anteced_i = np.argmax(old_scores)
            # new_scores = top_antecedent_scores[anaphor_span_idx][anteced_valid_arr_idxs] + scores
            # new_best_anteced_i = np.argmax(new_scores)
            # if new_best_anteced_i != prev_best_anteced_i:
            #     self._log_debug("*******************")
            #     self._log_debug("anaphor %d: (%d, %d) %s" % (anaphor_span_idx, anaphor_start, anaphor_end, " ".join(raw_bert_toks[anaphor_start:anaphor_end+1])))
            #     self._log_debug("  BEST ANTECED CHANGED:")
            #     self._log_debug("  stats: span_idx (start) str: s0_score/score_before/score_after")
            #     self._log_debug("  prev_best: %d (%d) %.2f/%.2f/%.2f %s\n[context] %s" % (
            #         all_anteced_valid_span_idxs[prev_best_anteced_i],
            #         all_anteced_starts[prev_best_anteced_i],
            #         old_scores[prev_best_anteced_i],
            #         scores[prev_best_anteced_i],
            #         new_scores[prev_best_anteced_i],
            #         all_anteced_strs[prev_best_anteced_i],
            #         all_input_strs[prev_best_anteced_i]
            #     ))
            #     self._log_debug("  new_best: %d (%d) %.2f/%.2f/%.2f %s\n[context] %s" % (
            #         all_anteced_valid_span_idxs[new_best_anteced_i],
            #         all_anteced_starts[new_best_anteced_i],
            #         old_scores[new_best_anteced_i],
            #         scores[new_best_anteced_i],
            #         new_scores[new_best_anteced_i],
            #         all_anteced_strs[new_best_anteced_i],
            #         all_input_strs[new_best_anteced_i]
            #     ))


        return top_antecedent_scores
