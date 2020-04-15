import torch
import json
import random
import numpy as np

from anagen.utils import flatten, combine_subtokens
from torch.utils.data import Dataset
from transformers import GPT2Tokenizer
from anagen.utils import invert_subtoken_map

from collections import Counter

GPT2_EOS_TOKEN_ID = 50256
MAX_NUM_SPEAKERS = 20

""" Stores tokenized strings and id form of a document used in the anaphor
    generation task."""
class AnagenDocument:
    def __init__(self, doc_key, segments, segment_starts, subtoken_map, speakers, tokenizer):
        self.doc_key = doc_key
        self.segment_toks = segments
        self.segment_ids = [tokenizer.encode(seg) for seg in segments]
        self.segment_starts = segment_starts
        self.subtoken_map = subtoken_map
        self.tokenizer = tokenizer

        if speakers is not None:
            self.raw_speakers = speakers
            speaker_dict = self.get_speaker_dict(speakers)
            self.speakers = [speaker_dict[s] for s in speakers]

    """ get speaker dict, copied from coref/independent.py CorefModel.get_speaker_dict() """
    def get_speaker_dict(self, speakers):
        speaker_dict = {'UNK': 0, '[SPL]': 1}
        for s in speakers:
            if s not in speaker_dict and len(speaker_dict) < MAX_NUM_SPEAKERS:
                speaker_dict[s] = len(speaker_dict)
        return speaker_dict

    """ Given indices ranging in the whole document, obtain corresponding
        segment and index in the segment"""
    def index_in_segments(self, start_idx, end_idx=None):
        # assume that no spans can cross segment boundaries
        seg_idx = np.argmax(self.segment_starts > start_idx) - 1
        start_idx_in_seg = start_idx - self.segment_starts[seg_idx]
        if end_idx is None:
            return seg_idx, start_idx_in_seg
        else:
            end_idx_in_seg = end_idx - self.segment_starts[seg_idx]
            return seg_idx, start_idx_in_seg, end_idx_in_seg

    """ Get the tokens of a span in the current document.
        Args:
            start (int): start idx
            end (int): end idx, inclusive
            in_segment (int or None): segment index, if given then the start
                and end above are relevant to this segment.
            output_str (bool): whether to undo bpe and return original string. """

    def decode(self, start, end, in_segment=None, output_str=True):
        if start == -1 and end == -1:
            return "<null>"
        if not in_segment:
            in_segment, in_seg_start, in_seg_end = self.index_in_segments(start, end)
            global_start, global_end = start, end
        else:
            global_start = self.segment_starts[in_segment] + start
            global_end = self.segment_starts[in_segment] + end
        span_toks = self.segment_toks[in_segment][in_seg_start:in_seg_end+1]
        subtoken_map = self.subtoken_map[global_start:global_end+1]

        res = combine_subtokens(span_toks, subtoken_map)

        if output_str:
            return " ".join(res)
        else:
            return res


""" Intermediate object representing one training example in the anaphor
    generation task"""
class AnagenExample:
    def __init__(self, doc_key, anteced_start, anteced_end, anaphor_start,
                 anaphor_end, ctx_seg_start_idx, ctx_seg_end_idx):
        self.doc_key = doc_key
        self.anteced_start = anteced_start
        self.anteced_end = anteced_end
        self.anaphor_start = anaphor_start
        self.anaphor_end = anaphor_end
        # indeces of segments that are used to
        self.ctx_seg_start_idx = ctx_seg_start_idx
        self.ctx_seg_end_idx = ctx_seg_end_idx

    def __str__(self):
        return "(Doc \"%s\", segs %d-%d, anteced (%d, %d), anaphor (%d, %d))" \
            % (self.doc_key, self.ctx_seg_start_idx, self.ctx_seg_end_idx,
               self.anteced_start, self.anteced_end,
               self.anaphor_start, self.anaphor_end)


""" Dataset class to input examples for literal speaker of anaphor generation
    task. Takes jsonlines file constructed by anagen_minimize.py as input.
    Initialization:
    [Step 1] _process_jsonline(): reads all lines in the jsonlines file and
        constructs examples and document objects
    [Step 2] _finalize_batches: given example and document objects, construct
        batches of a given batch size. Stores batches in memory

    Sampling batches:
    [Step 1] __getitem__(): takes a batch in memory and retrieves full id form
        of all necessary context sequences. After this step, a batch does not
        contain any reference to document objects.
    [Step 2] collate(batch): converts everything to PyTorch tensors; reorganizes
        examples in order of decreasing anaphor length; adds padding to context
        and anaphor sequences. """
class AnagenDataset(Dataset):
    def __init__(self, input_file=None, data_augment=None, data_augment_file=None,
                 batch_size=32, max_span_width=10, data_augment_max_span_width=10,
                 max_num_ctxs_in_batch=8, max_segment_len=256,
                 use_speaker_info=False, shuffle=False, tokenizer=None):
        self.documents = {}
        self.docs_to_examples = {}
        self.batches = []
        self.batch_size = batch_size
        self.max_span_width = max_span_width
        self.data_augment_max_span_width = data_augment_max_span_width
        self.max_num_ctxs_in_batch = max_num_ctxs_in_batch
        self.max_segment_len = max_segment_len
        self.use_speaker_info = use_speaker_info
        self.shuffle = shuffle

        self.num_examples = 0
        if tokenizer:
            self.tokenizer = tokenizer
        else:
            self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

        # just use print for debugging and logging purposes now
        print("Initializing dataset from %s" % input_file)
        print("Shuffling examples in each document" if self.shuffle else "Not shuffling examples in document")
        if data_augment is None:
            with open(input_file, "r") as f:
                for line in f:
                    self._process_example(json.loads(line))
        else:
            if data_augment == "null_from_l0":
                aug_f = open(data_augment_file, "rb")
                data_f = open(input_file, "r")

                data_dicts = np.load(aug_f, allow_pickle=True).item().get("data_dicts")
                for line_num, line in enumerate(data_f):
                    self._process_example(json.loads(line), aug_data_dict=data_dicts[line_num])
                data_f.close()
                aug_f.close()
            else:
                raise NotImplementedError()

        # analysis of sizes
        # null_anaphor_lens = [ex.anaphor_end - ex.anaphor_start + 1 for exs in self.docs_to_examples.values() for ex in exs if ex.anteced_start == -1]
        # null_anaphor_len_distr = Counter(null_anaphor_lens)
        # print("lens of anaphors with null antecedents", null_anaphor_len_distr.most_common())
        # avg_anaphor_len = sum([ex.anaphor_end-ex.anaphor_start + 1 for exs in self.docs_to_examples.values() for ex in exs]) / self.num_examples
        # print("avg anaphor len %.2f" % avg_anaphor_len)

        self.num_null_examples = len([None for exs in self.docs_to_examples.values() for ex in exs if ex.anteced_start == -1])

        print("Obtained %d examples, %d (%.2f%%) examples with null antecedents" \
              % (self.num_examples, self.num_null_examples, self.num_null_examples/self.num_examples*100))
        print("Compiling batches, batch size %d..." % self.batch_size)
        self._finalize_batches()
        print("Compiled %d batches." % len(self.batches))

        num_examples_in_batches = 0
        for b in self.batches:
            num_examples_in_batches += len(b[2])

    """ Get the tokens of a span in a given document.
        See definition in AnagenDocument.decode()"""
    def decode(self, doc, start, end, in_segment=None, output_str=True):
        if start == -1 and end == -1:
            return "<null>"
        if isinstance(doc, str):
            return self.documents[doc].decode(start, end, in_segment, output_str)
        if isinstance(doc, AnagenDocument):
            return doc.decode(start, end, in_segment, output_str)
        else:
            ids = doc[start:end+1]
            return self.decode_ids(ids)

    def decode_ids(self, ids):
        # TODO: add function to revert to original word form using subtoken_map.
        if torch.is_tensor(ids):
            ids = ids.tolist()
        return self.tokenizer.decode(ids)


    """ Go through jsonlines file and obtain training examples for anaphor generation"""
    def _process_example(self, coref_example, aug_data_dict=None):
        doc_key = coref_example["doc_key"]
        segments = coref_example["sentences"]
        speakers = flatten(coref_example["speakers"])
        subtoken_map = coref_example["subtoken_map"]
        clusters = coref_example["clusters"]

        # includes one extra element to facilitate _get_ctx_seg_idxs().
        segment_lens = np.array([0] + [len(s) for s in segments])
        segment_starts = np.cumsum(segment_lens)

        document = AnagenDocument(doc_key, segments, segment_starts,
                                  subtoken_map, speakers, self.tokenizer)
        self.documents[doc_key] = document

        # get cluster information
        anagen_examples = []
        anaphors_with_null_anteceds = set()
        for cluster in clusters:
            mentions = sorted(tuple(m) for m in cluster)
            for anaphor_i in range(len(mentions)):
                anaphor_start = mentions[anaphor_i][0]
                anaphor_end = mentions[anaphor_i][1]

                if self.max_span_width is not None \
                    and anaphor_end - anaphor_start + 1 > self.max_span_width:
                    continue

                ctx_seg_start_idx, ctx_seg_end_idx = \
                    self.get_ctx_seg_idxs(segment_starts, anaphor_start)
                if anaphor_i == 0:
                    # first mention
                    anaphors_with_null_anteceds.add((anaphor_start, anaphor_end))
                    # add to list of examples later
                else:
                    for anteced_i in range(anaphor_i):
                        anteced_start = mentions[anteced_i][0]
                        anteced_end = mentions[anteced_i][1]
                        if anteced_end - anteced_start + 1 > self.max_span_width \
                            or anteced_end >= anaphor_start:
                            continue
                        ex = AnagenExample(doc_key,
                                           anteced_start, anteced_end,
                                           anaphor_start, anaphor_end,
                                           ctx_seg_start_idx, ctx_seg_end_idx)
                        anagen_examples.append(ex)

        # print("num of clusters", len(clusters))
        # print("num of cluster first mentions", len(anaphors_with_null_anteceds))
        # NOTE: there seem to be some clusters containing the same span as first
        # mention, i.e. num first mentions < num clusters.

        # test data dict
        if aug_data_dict:
            gpt_word_to_subtok_start, gpt_word_to_subtok_end = invert_subtoken_map(subtoken_map)
            gpt_word_to_subtok_start = np.array(gpt_word_to_subtok_start)
            gpt_word_to_subtok_end = np.array(gpt_word_to_subtok_end)
            bert_subtok_to_word = np.array(aug_data_dict["example"]["subtoken_map"])
            null_anteceds = np.argmax(aug_data_dict["top_antecedent_scores"], axis=1) == 0
            bert_starts = aug_data_dict["top_span_starts"][null_anteceds]
            bert_ends = aug_data_dict["top_span_ends"][null_anteceds]
            word_starts = bert_subtok_to_word[bert_starts]
            word_ends = bert_subtok_to_word[bert_ends]
            # print("word_starts.shape", word_starts.shape)
            # print("word_ends.shape", word_ends.shape)
            gpt_starts = gpt_word_to_subtok_start[word_starts]
            gpt_ends = gpt_word_to_subtok_end[word_ends]

            # print("gpt_ends.shape", gpt_ends.shape)
            # print("gpt_starts[:10]", gpt_starts[:10])
            # print("gpt_ends[:10]", gpt_ends[:10])
            # for start, end in zip(gpt_starts, gpt_ends):
            #     print(document.decode(start, end))
            anaphors_with_null_anteceds.update(zip(gpt_starts, gpt_ends))

        for anaphor_start, anaphor_end in anaphors_with_null_anteceds:
            # IMPORTANT: may cap max span width for null anaphors at
            if anaphor_end - anaphor_start + 1 > self.data_augment_max_span_width:
                continue
            ctx_seg_start_idx, ctx_seg_end_idx = \
                self.get_ctx_seg_idxs(segment_starts, anaphor_start)
            ex = AnagenExample(doc_key,
                               -1, -1,
                               anaphor_start, anaphor_end,
                               ctx_seg_start_idx, ctx_seg_end_idx)
            anagen_examples.append(ex)

        if self.shuffle:
            random.shuffle(anagen_examples)

        self.docs_to_examples[doc_key] = anagen_examples
        self.num_examples += len(anagen_examples)


    """ Determine the start of the context"""
    def get_ctx_seg_idxs(self, segment_starts, anaphor_start, max_segment_len=None):
        max_segment_len = self.max_segment_len if max_segment_len is None else max_segment_len

        ctx_seg_start_idx = np.argmax(anaphor_start - segment_starts <= max_segment_len)
        ctx_seg_end_idx = np.argmax(segment_starts > anaphor_start) - 1
        # print("anaphor_start", anaphor_start, "segment_starts", segment_starts, "start_idx", ctx_seg_start_idx)
        return ctx_seg_start_idx, ctx_seg_end_idx

    """ After preparing all examples, group them into batches stored in memory"""
    def _finalize_batches(self):
        for doc_key, examples in self.docs_to_examples.items():
            # ensure all examples in a batch are from the same document
            curr_batch = []
            ctxs = set()
            for example in examples:
                ctxs.add(example.ctx_seg_start_idx)
                if len(curr_batch) >= self.batch_size or len(ctxs) > self.max_num_ctxs_in_batch:
                    self._columnize_and_add_batch(curr_batch, doc_key)
                    curr_batch = [example]
                    ctxs = set([example.ctx_seg_start_idx])
                else:
                    curr_batch.append(example)
            if len(curr_batch) > 0:
                self._columnize_and_add_batch(curr_batch, doc_key)


    """convert batch to "column" form and append batch to self.batches"""
    def _columnize_and_add_batch(self, batch, doc_key):
        # print("***adding a new batch***")
        doc = self.documents[doc_key]
        # print("ctx_seg_start_idxs", [ex.ctx_seg_start_idx for ex in batch])
        ctx_starts = [doc.segment_starts[ex.ctx_seg_start_idx] for ex in batch]

        anteced_starts = [ex.anteced_start for ex in batch]
        anteced_ends = [ex.anteced_end for ex in batch]
        anaphor_starts = [ex.anaphor_start for ex in batch]
        # print("ctx_starts", ctx_starts)
        # print("anaphor_starts", anaphor_starts)
        if self.use_speaker_info:
            speaker_info = [doc.speakers[ex.anteced_start] == doc.speakers[ex.anaphor_start] \
                            if ex.anteced_start >= 0 else False
                            for ex in batch]

        # prepare anaphors in id form, and prepare ctx_set:
        # for all ctx ranges (denoted by tuples) in batch, remove duplicates to
        # get a set, and for ranges with same start idx, keep only the longest
        # range.
        all_anaphor_ids = []
        ctxs = [None for _ in range(len(doc.segment_toks))]
        for ex in batch:
            # print(ex)
            # obtain id form of anaphor, store in memory
            anaphor_seg_idx, start_idx_in_seg, end_idx_in_seg = \
                doc.index_in_segments(ex.anaphor_start, ex.anaphor_end)

            anaphor_ids = doc.segment_ids[anaphor_seg_idx][start_idx_in_seg:end_idx_in_seg+1]
            if isinstance(anaphor_ids, int):
                anaphor_ids = [anaphor_ids]
            # if len(batch) == 1:
            #     print("segment_starts", doc.segment_starts)
            #     print("anaphor_seg_idx = %d, start_idx_in_seg = %d, end_idx_in_seg = %d" % (anaphor_seg_idx, start_idx_in_seg, end_idx_in_seg))
            #     print(anaphor_id)
            all_anaphor_ids.append(anaphor_ids)

            ctx_seg_start_idx, ctx_seg_end_idx = ex.ctx_seg_start_idx, ex.ctx_seg_end_idx
            if ctxs[ctx_seg_start_idx] is None or ctxs[ctx_seg_start_idx][1] < ctx_seg_end_idx:
                ctxs[ctx_seg_start_idx] = (ctx_seg_start_idx, ctx_seg_end_idx)
        ctx_set = [ctx for ctx in ctxs if ctx is not None]
        start_idx_to_set_idx = {
            ctx_seg_start_idx: i \
            for i, ctx_seg_start_idx \
            in enumerate([start_i for (start_i, _) in ctx_set])
        }

        # for each item in the batch, get the index of the corresponding context
        # in the set of contexts
        ctx_set_idxs = [start_idx_to_set_idx[ex.ctx_seg_start_idx] for ex in batch]

        # print("doc.segment_starts", doc.segment_starts) #ok
        # print("ctx_set", ctx_set, "ctx_set_idxs", ctx_set_idxs, "ctx_starts", ctx_starts) #ok
        # print("anteced_starts", anteced_starts) #ok
        # print("anteced_ends", anteced_ends) #ok
        # print("anaphor_ids", anaphor_ids) #ok
        #
        # for s, e, anaphor_id in zip(anteced_starts, anteced_ends, anaphor_ids):
            # print("[anteced]", self.decode(doc, s, e), "[anaphor]", self.decode(anaphor_id))

        if self.use_speaker_info:
            batch_tuple = (doc_key, ctx_set, ctx_starts, ctx_set_idxs,
                           anteced_starts, anteced_ends, anaphor_starts, all_anaphor_ids,
                           speaker_info)
        else:
            batch_tuple = (doc_key, ctx_set, ctx_starts, ctx_set_idxs,
                           anteced_starts, anteced_ends, anaphor_starts, all_anaphor_ids)
        self.batches.append(batch_tuple)

    def __len__(self):
        return len(self.batches)

    """ obtain full id form of ctx, pass on everything else """
    def __getitem__(self, idx):
        if self.use_speaker_info:
            doc_key, ctx_set, ctx_starts, ctx_set_idxs, anteced_starts, anteced_ends, \
                 anaphor_starts, anaphor_ids, speaker_info = self.batches[idx]
        else:
            doc_key, ctx_set, ctx_starts, ctx_set_idxs, anteced_starts, anteced_ends, \
                 anaphor_starts, anaphor_ids = self.batches[idx]
            speaker_info = None

        doc = self.documents[doc_key]

        # obtain full id form of ctx
        ctx_ids = [flatten(doc.segment_ids[start_i:end_i+1]) for start_i, end_i in ctx_set]

        # for s, e, anaphor_id in zip(anteced_starts_in_ctx, anteced_ends_in_ctx, anaphor_ids):
            # print("[anteced]", self.decode(doc, s, e), "[anaphor]", self.decode(anaphor_id))

        return doc_key, ctx_set, ctx_starts, ctx_ids, ctx_set_idxs, \
            anteced_starts, anteced_ends, anaphor_starts, anaphor_ids, speaker_info

""" Processes data retrieved from Dataset.__getitem__, converts everything into
    tensors. Returns a dictionary containing content of batch:
    'ctx_ids': [num_ctxs, max_ctx_len] id form of context tokens
    'ctx_ids_padding_mask': [num_ctxs, max_ctx_len] mask indicating length
        variation in ctx, to pass into GPT2 model
    'ctx_lens': [num_ctxs,] length of ctxs
    'ctx_starts': [batch_size,] starting idx of ctx for eachg example
    'ctx_set_idxs': [batch_size,] the corresponding ctx for each example
    'anteced_starts': [batch_size,] starting idxs of antecedents, indexed
        into each respective corresponding ctx
    'anteced_ends': [batch_size,] end idxs of antecedents
    'anaphor_starts': [batch_size,] starting idxs anaphors
    'anaphor_ids': [batch_size, max_anaphor_len] id form of gold anaphor tokens
    'anaphor_ids_padding_mask': [batch_size, max_anaphor_len] mask
        indicating length variation in ctx, to pass into GPT2 model
    'anaphor_lens': [batch_size,] lengths of gold anpahors
    'scramble_idxs': [batch_size,] tensor of indices that is applied to sort
        examples in order of decreasing anaphor length."""
def collate(batch):
    doc_key, _, ctx_starts, ctx_ids, ctx_set_idxs, anteced_starts, anteced_ends, \
        anaphor_starts, anaphor_ids, speaker_info = batch[0]

    # transform everything else into tensor form.
    # All following tensors have dim [batch_size,] unless noted
    ctx_lens = torch.tensor([len(ctx_id) for ctx_id in ctx_ids]) # [num_ctxs,]
    ctx_starts = torch.tensor(ctx_starts)
    ctx_set_idxs = torch.tensor(ctx_set_idxs)
    anteced_starts = torch.tensor(anteced_starts)
    anteced_ends = torch.tensor(anteced_ends)
    anaphor_starts = torch.tensor(anaphor_starts)
    speaker_info = torch.tensor(speaker_info).long() if speaker_info is not None else None
    anteced_starts_in_ctx = torch.clamp(anteced_starts - ctx_starts, min=-1)
    anteced_ends_in_ctx = torch.clamp(anteced_ends - ctx_starts, min=-1)
    anaphor_starts_in_ctx = anaphor_starts - ctx_starts

    # just pad ctxs, don't sort them
    ctx_ids = [torch.tensor(ctx) for ctx in ctx_ids] # convert to tensor form
    padded_ctx_ids = torch.nn.utils.rnn.pad_sequence(ctx_ids, batch_first=True,
                                                     padding_value=GPT2_EOS_TOKEN_ID)
    ctx_ids_padding_mask = (torch.arange(padded_ctx_ids.shape[1])[None, :] \
                            < ctx_lens[:, None]).type(torch.FloatTensor)

    # prepare anaphor ids
    anaphor_lens = torch.tensor([len(anaphor_id) for anaphor_id in anaphor_ids])
    sorted_anaphor_lens, scramble_idxs = torch.sort(anaphor_lens, descending=True)
    sorted_anaphor_ids = [torch.tensor(anaphor_ids[i]) for i in scramble_idxs]
    padded_anaphor_ids = torch.nn.utils.rnn.pad_sequence(sorted_anaphor_ids, batch_first=True,
                                                         padding_value=GPT2_EOS_TOKEN_ID)
    anaphor_ids_padding_mask = (torch.arange(padded_anaphor_ids.shape[1])[None, :] \
                                < sorted_anaphor_lens[:, None]).type(torch.FloatTensor)
    # probably don't need this one

    # prepare anteced
    sorted_ctx_starts = ctx_starts[scramble_idxs]
    sorted_ctx_set_idxs = ctx_set_idxs[scramble_idxs]
    sorted_anteced_starts = anteced_starts_in_ctx[scramble_idxs]
    sorted_anteced_ends = anteced_ends_in_ctx[scramble_idxs]
    sorted_anaphor_starts = anaphor_starts_in_ctx[scramble_idxs]
    sorted_speaker_info = speaker_info[scramble_idxs] if speaker_info is not None else None

    batch_dict = {
        'doc_key': doc_key,
        'ctx_ids': padded_ctx_ids,
        'ctx_lens': ctx_lens,
        'ctx_ids_padding_mask': ctx_ids_padding_mask,
        'ctx_starts': sorted_ctx_starts,
        'ctx_set_idxs': sorted_ctx_set_idxs,
        'anteced_starts': sorted_anteced_starts,
        'anteced_ends': sorted_anteced_ends,
        'anaphor_starts': sorted_anaphor_starts,
        'anaphor_ids': padded_anaphor_ids,
        'anaphor_ids_padding_mask': anaphor_ids_padding_mask,
        'anaphor_lens': sorted_anaphor_lens,
        'scramble_idxs': scramble_idxs,
        'speaker_info': sorted_speaker_info
    }

    return batch_dict
