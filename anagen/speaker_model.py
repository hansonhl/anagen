import torch
import math
from torch import nn
from anagen.dataset import GPT2_EOS_TOKEN_ID
from transformers import GPT2Model, GPT2Tokenizer

debug_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

class RNNSpeakerModel(nn.Module):
    def __init__(self, args, device=None):
        super(RNNSpeakerModel, self).__init__()
        if device:
            self.device=device
        else:
            self.device = torch.device("cuda" if args.gpu else "cpu")
        # some previous versions do not have metadata options in args
        if hasattr(args, "use_speaker_info"):
            self.use_speaker_info = args.use_speaker_info
            self.metadata_emb_size = args.metadata_emb_size
        else:
            self.use_speaker_info = False
        if hasattr(args, "use_distance_info"):
            self.use_distance_info = args.use_distance_info
            self.distance_groups = args.distance_groups
        else:
            self.use_distance_info = False

        self.sum_start_end_emb = args.sum_start_end_emb

        self.gpt2_hidden_size = args.gpt2_hidden_size
        self.anteced_emb_size = args.gpt2_hidden_size if args.sum_start_end_emb\
                                else args.gpt2_hidden_size * 2
        self.ctx_emb_size = args.gpt2_hidden_size
        self.rnn_hidden_size = self.anteced_emb_size + self.ctx_emb_size

        if self.use_speaker_info:
            self.rnn_hidden_size += self.metadata_emb_size
        if self.use_distance_info:
            self.rnn_hidden_size += self.metadata_emb_size
        print("self.rnn_hidden_size", self.rnn_hidden_size)

        self.gpt2_model = GPT2Model.from_pretrained(args.gpt2_model_dir \
            if args.gpt2_model_dir else "gpt2")

        self.vocab_size = self.gpt2_model.wte.num_embeddings

        self.rnn = nn.GRU(input_size=args.gpt2_hidden_size,
                          hidden_size=self.rnn_hidden_size,
                          num_layers=args.rnn_num_layers,
                          batch_first=True)
        self.hidden_to_logits = nn.Linear(self.rnn_hidden_size, self.vocab_size)
        self.null_anteced_emb = nn.Parameter(args.param_init_stdev * torch.randn(self.gpt2_hidden_size))
        self.anaphor_start_emb = nn.Parameter(args.param_init_stdev * torch.randn(self.gpt2_hidden_size))

        if self.use_speaker_info:
            self.speaker_emb = nn.Embedding(2, self.metadata_emb_size)
        if self.use_distance_info:
            # use 32 even buckets of width 16, last bucket is for everything else
            self.distance_emb = nn.Embedding(self.distance_groups, self.metadata_emb_size)

        self.end_tok = torch.tensor([GPT2_EOS_TOKEN_ID], device=self.device, requires_grad=False)
        self.loss_fxn = nn.CrossEntropyLoss()

    @classmethod
    def from_checkpoint(cls, checkpoint_path, device=None):
        ckpt = torch.load(checkpoint_path)
        args = ckpt["args"]
        model = cls(args, device)
        model.load_state_dict(ckpt["model_state_dict"])
        return model

    def freeze_gpt2(self):
        for param in self.gpt2_model.parameters():
            param.requires_grad = False
        self.gpt2_model.eval()

    def unfreeze_gpt2(self):
        for param in self.gpt2_model.parameters():
            param.requires_grad = True

    def token_embedding(self, anaphor_ids):
        return self.gpt2_model.wte(anaphor_ids)

    def forward(self, batch, verbose=False):
        input_repr_embs = self.encode(batch, verbose) # [batch_size, gpt2_hidden_size * 3]
        logits = self.decode(input_repr_embs, batch["anaphor_ids"]) # [batch_size, max_len+1, vocab_size]

        return {
            "logits": logits,
            "loss": self.loss(logits, batch["anaphor_ids"], batch["anaphor_ids_padding_mask"]),
            "num_toks": self.num_toks(batch["anaphor_ids_padding_mask"])
        }

    def encode(self, batch, verbose=False):
        ctx_ids = batch["ctx_ids"] # [num_ctx, max_ctx_len]
        ctx_ids_padding_mask = batch["ctx_ids_padding_mask"] # [num_ctx, max_ctx_len]
        ctx_set_idxs = batch["ctx_set_idxs"] # [batch_size,]
        anteced_starts = batch["anteced_starts"] # [batch_size,]
        anteced_ends = batch["anteced_ends"] # [batch_size,]
        anaphor_starts = batch["anaphor_starts"] # [batch_size,]
        speaker_info = batch["speaker_info"] # [batch_size,]

        gpt2_output = self.gpt2_model(ctx_ids, attention_mask=ctx_ids_padding_mask)
        hidden_states = gpt2_output[0]
        if verbose:
            print("hidden_states.shape", hidden_states.shape)
            print("hidden_states", hidden_states)
        # [num_ctxs, max_ctx_len, gpt2_hidden_size]

        # flatten everything and prepend null representation to faciliate index selection
        flat_hidden_states = torch.cat((self.null_anteced_emb.unsqueeze(0),
            hidden_states.view(-1, hidden_states.shape[-1])),0)
        # [1+num_ctxs*max_ctx_len, gpt2_hidden_size]

        emb_list = []

        # get antecedent embeddings
        null_anteced = anteced_starts == -1
        flat_idx_offset = (torch.arange(ctx_ids.shape[0], device=self.device) \
                           * ctx_ids.shape[1])[ctx_set_idxs] # [batch_size,]
        flat_anteced_start_idxs = flat_idx_offset + anteced_starts + 1 # [batch_size,]
        flat_anteced_end_idxs = flat_idx_offset + anteced_ends + 1 # [batch_size,]
        flat_anteced_start_idxs[null_anteced] = 0
        flat_anteced_end_idxs[null_anteced] = 0
        anteced_start_embs = flat_hidden_states.index_select(0, flat_anteced_start_idxs)
        anteced_end_embs = flat_hidden_states.index_select(0, flat_anteced_end_idxs)
        # [batch_size, gpt2_hidden_size]
        if self.sum_start_end_emb:
            emb_list.append(anteced_start_embs + anteced_end_embs)
        else:
            emb_list.append(anteced_start_embs)
            emb_list.append(anteced_end_embs)

        # get context embeddings
        # just use emb of one token before anaphor
        # if anaphor has index 0, use null anteced emb as ctx emb
        flat_ctx_ends = flat_idx_offset + anaphor_starts # [batch_size, ]
        ctx_embs = flat_hidden_states.index_select(0, flat_ctx_ends)
        # [batch_size, gpt2_hidden_size]
        emb_list.append(ctx_embs)

        if self.use_speaker_info:
            speaker_embs = self.speaker_emb(speaker_info)
            emb_list.append(speaker_embs)
        if self.use_distance_info:
            distances = anaphor_starts - anteced_starts
            distance_ids = torch.clamp(distances / 16, min=0, max=self.distance_groups-1)
            distance_embs = self.distance_emb(distance_ids)
            emb_list.append(distance_embs)

        input_embs = torch.cat(emb_list, 1)
        # [batch_size, gpt2_hidden_size * 3]
        return input_embs

    def decode(self, ctx_and_anteced_embs, prev_anaphor_ids):
        # input_repr_embs:
        # anaphor_ids: [batch_size, max_anaphor_len]
        ctx_and_anteced_embs = ctx_and_anteced_embs.unsqueeze(0) # [1, batch_size, rnn_hidden_size]
        anaphor_embs = self.token_embedding(prev_anaphor_ids) # [batch_size, max_len, gpt2_hidden_size]

        # append start token embedding to each example in batch
        anaphor_start_embs = self.anaphor_start_emb[None, None, :] # [1, 1, gpt2_hidden_size]
        anaphor_start_embs = anaphor_start_embs.repeat(prev_anaphor_ids.shape[0], 1, 1) # [batch_size, 1, gpt2_hidden_size]
        anaphor_embs = torch.cat((anaphor_start_embs, anaphor_embs), 1) # [batch_size, max_len+1, gpt2_hidden_size]

        # run rnn
        rnn_hidden_states, _ = self.rnn(anaphor_embs, ctx_and_anteced_embs) # [batch_size, max_len+1, rnn_hidden_size]
        scores = self.hidden_to_logits(rnn_hidden_states) # [batch_size, max_len+1, vocab_size]

        return scores

    def loss(self, logits, anaphor_ids, mask):
        # logits: [batch_size, max_anaphor_len+1, vocab_size]
        # anaphor_ids, mask: [batch_size, max_anaphor_len]
        batch_size = logits.shape[0]

        # append end of sentence token to gold ids
        end_toks = self.end_tok.unsqueeze(0).repeat(batch_size, 1)
        gold_anaphor_ids = torch.cat((anaphor_ids, end_toks), 1) # [batch_size, max_len+1]

        # adjust mask to include start token
        mask = torch.cat((torch.ones(batch_size, 1, device=self.device), mask), 1).bool()

        # flatten and filter everything
        flat_mask = mask.view(-1)
        logits = logits.view(-1, logits.shape[-1])[flat_mask]
        gold_anaphor_ids = gold_anaphor_ids.view(-1)[flat_mask]

        loss = self.loss_fxn(logits, gold_anaphor_ids)
        return loss

    def num_toks(self, mask):
        batch_size = mask.shape[0]
        # adjust mask to include start token
        mask = torch.cat((torch.ones(batch_size, 1, device=self.device), mask), 1).bool()
        return mask.sum()
