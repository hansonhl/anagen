import torch
import logging
import tqdm
import time
from anagen.dataset import collate
from anagen.utils import batch_to_device

from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler

DEFAULT_TRAIN_BATCH_SIZE=1
logger = logging.getLogger(__name__)

def parse_train_args(parser):
    # data input
    parser.add_argument("--train_jsonlines", type=str)
    parser.add_argument("--eval_jsonlines", type=str)
    parser.add_argument("--train_batch_size", type=int, default=16)
    parser.add_argument("--eval_batch_size", type=int, default=16)
    parser.add_argument("--max_span_width", type=int, default=10)
    parser.add_argument("--max_num_ctxs_in_batch", type=int, default=8)
    parser.add_argument("--max_segment_len", type=int, default=512)

    # trained model save/load
    parser.add_argument("--model_load_path", type=str)
    parser.add_argument("--model_save_path", type=str)

    # gpt2 model settings
    parser.add_argument("--gpt2_model_dir", type=str, default=None)

    # training settings
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--random_seed", type=int, default=39393)
    parser.add_argument("--unfreeze_gpt2", action="store_true")
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--train_epochs", type=int, default=1)
    parser.add_argument("--log_steps", type=int, default=100)
    parser.add_argument("--eval_and_save_steps", type=int, default=5000)

    # model settings
    parser.add_argument("--gpt2_hidden_size", type=int, default=768)
    parser.add_argument("--stack_start_end_emb", action="store_true")
    parser.add_argument("--use_metadata", action="store_true")
    parser.add_argument("--param_init_stdev", type=float, default=0.1)
    parser.add_argument("--rnn_num_layers", type=int, default=1)

    return parser.parse_args()


def check_state_dict(model, optimizer=None):
    print("****** checking state dict of model ******")
    # print(state_dict.keys())
    speaker_state_dict = model.state_dict()
    gpt2_state_dict = model.gpt2_model.state_dict()
        # print("optimizer.state_dict()['param_groups']", optimizer.state_dict()['param_groups'])

    print("gpt2_model.wte.requires_grad", model.gpt2_model.wte.weight.requires_grad)
    print("null_anteced_emb.requires_grad", model.null_anteced_emb.requires_grad)
    print("gpt2_model.wte.weight.grad", model.gpt2_model.wte.weight.grad)
    return (gpt2_state_dict["h.0.attn.bias"][0][0][0][:10].tolist(),
            gpt2_state_dict["wte.weight"][9][:10].tolist(),
            speaker_state_dict["token_embedding.weight"][9][:10].tolist(),
            model.null_anteced_emb.data[:10].tolist(),
            model.hidden_to_logits.weight[0][:10].tolist(),
            )


# based on transformers/run_lm_finetuning
def train(args, model, train_dataset, eval_dataset):
    device = torch.device("cuda" if args.gpu else "cpu")
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler,
                                  batch_size=DEFAULT_TRAIN_BATCH_SIZE,
                                  collate_fn=collate)

    # not sure what the following does, copied from run_lm_finetuning code
    """
    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps \
            // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // \
            args.gradient_accumulation_steps * args.num_train_epochs
    """
    model.to(device)

    global_step = 0

    gpt_bias1, gpt_wte1, s0_emb1, null_emb1, s0_h2l1 = check_state_dict(model)

    if args.model_load_path:
        # Load in model and optimizer states
        print("***** Loading model from %s *****" % args.model_load_path)
        ckpt = torch.load(args.model_load_path)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        global_step = ckpt["global_step"]
        print("***** Finished loading model *****")

    if not args.unfreeze_gpt2:
        model.freeze_gpt2()
    else:
        print("***** Unfreezing gpt2 parameters *****")
        model.unfreeze_gpt2()

    print("***** List of all parameters: *****")
    for name, param in model.named_parameters():
        print("  %s %s %s" % ("[GRAD]" if param.requires_grad else "[NONE]", name, param.shape))
    print("***** End of list *****")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    gpt_bias2, gpt_wte2, s0_emb2, null_emb2, s0_h2l2 = check_state_dict(model, optimizer)

    num_batches = len(train_dataset)
    # start training
    print("***** Running training *****")
    print("  Num examples = %d" % num_batches)
    print("  Num Epochs = %d" % args.train_epochs)
    print("  Batch size = %d" % args.train_batch_size)
    print("  Logging every %d steps" % args.log_steps)
    print("  Evaluating and saving model every %d steps" % args.eval_and_save_steps)

    if args.model_save_path:
        # keep track of best loss for early stopping
        best_loss = float("inf")
        best_step = 0

    total_training_time = 0.0
    for epoch in range(args.train_epochs):
        print("*** Epoch %d ***" % epoch)
        for step, batch in enumerate(train_dataloader):
            batch = batch_to_device(batch, device)
            model.zero_grad()
            model.train()

            start_time = time.time()
            res_dict = model(batch)
            loss = res_dict["loss"]

            loss.backward()
            if global_step % args.log_steps == 0:
                print("gpt2_model.wte.weight.grad", model.gpt2_model.wte.weight.grad)
            optimizer.step()
            total_training_time += time.time() - start_time
            global_step += 1

            loss = loss.item()
            del res_dict

            if global_step % args.log_steps == 0:
                avg_time_per_batch = total_training_time / global_step
                estimated_time = (num_batches - (step+1)) * avg_time_per_batch
                print("  step %d/%d, global_step %d, batch loss = %.6f" \
                      % (step+1, num_batches, global_step, loss))
                print("  avg time per batch = %.2f, est remaining time = %.2f mins" \
                      % (avg_time_per_batch, estimated_time / 60))

            if global_step % args.eval_and_save_steps == 0:
                eval_results = evaluate(args, model, eval_dataset, global_step)
                # TODO: add tensorboard writer functionality
                eval_loss = eval_results["eval_loss"]

                if args.model_save_path:
                    model_checkpoint = {
                        "args": args,
                        "epoch": epoch,
                        "eval_loss": eval_loss,
                        "step_in_epoch": step,
                        "global_step": global_step,
                        "model_state_dict": model.state_dict(), # just save everything for now
                        "optimizer_state_dict": optimizer.state_dict()
                    }
                    if eval_loss < best_loss:
                        best_save_path = args.model_save_path + "_best.ckpt"
                        print("  current model has best eval loss, saving to %s" % best_save_path)
                        torch.save(model_checkpoint, best_save_path)
                        best_loss = eval_loss

                    latest_save_path = args.model_save_path + "_latest.ckpt"
                    print("  saving latest version of model to %s" % latest_save_path)
                    torch.save(model_checkpoint, latest_save_path)
                    del model_checkpoint

    gpt_bias3, gpt_wte3, s0_emb3, null_emb3, s0_h2l3 = check_state_dict(model)
    print("compare gpt_bias", gpt_bias1 == gpt_bias3)
    print("compare gpt_wte", gpt_wte1 == gpt_wte3)
    print("compare s0_emb", s0_emb1 == s0_emb3)
    print("compare null_emb", null_emb1 == null_emb3)
    print("compare s0_h2l", s0_h2l1 == s0_h2l3)

def evaluate(args, model, eval_dataset, global_step):
    device = torch.device("cuda" if args.gpu else "cpu")
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler,
                                 batch_size=DEFAULT_TRAIN_BATCH_SIZE,
                                 collate_fn=collate)

    print("***** Running evaluation at step %d ****" % global_step)
    print("  Num examples = %d" % len(eval_dataset))
    print("  Batch size = %d" % args.eval_batch_size)

    eval_loss = 0.0
    num_toks = 0
    model.eval()

    for step, batch in enumerate(eval_dataloader):
        with torch.no_grad():
            batch = batch_to_device(batch, device)
            res_dict = model(batch)
            if step + 1 % args.log_steps == 0:
                print("  evaluated %d batches" % step + 1)
            eval_loss += res_dict["loss"].item() * res_dict["num_toks"].item()
            num_toks += res_dict["num_toks"].item()
            del res_dict

    eval_loss = eval_loss / num_toks
    perplexity = torch.exp(torch.tensor(eval_loss))

    print("***** Eval results at step %d  *****" % global_step)
    print("  eval_loss = %.6f" % eval_loss)
    print("  perplexity = %.6f" % perplexity)
    return {
        "eval_loss": eval_loss,
        "perplexity": perplexity
    }
