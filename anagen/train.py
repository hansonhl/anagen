import torch
import logging
import tqdm
import time
import math
from anagen.dataset import collate
from anagen.utils import batch_to_device

from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler

DEFAULT_TRAIN_BATCH_SIZE=1
logger = logging.getLogger(__name__)

def parse_train_args(parser):
    # data input
    parser.add_argument("--train_input_file", type=str)
    parser.add_argument("--eval_input_file", type=str)
    parser.add_argument("--train_batch_size", type=int, default=16)
    parser.add_argument("--eval_batch_size", type=int, default=16)
    parser.add_argument("--max_span_width", type=int, default=10)
    parser.add_argument("--max_num_ctxs_in_batch", type=int, default=8)
    parser.add_argument("--max_segment_len", type=int, default=256)
    parser.add_argument("--shuffle_examples", action="store_true")
    parser.add_argument("--data_augment", type=str, choices=[None, "null_from_l0"])
    parser.add_argument("--data_augment_max_span_width", type=int, default=10)
    parser.add_argument("--train_data_augment_file", type=str)
    parser.add_argument("--eval_data_augment_file", type=str)

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
    parser.add_argument("--eval_and_save_by_epoch", type=float)
    parser.add_argument("--save_optimizer_state", action="store_true")
    parser.add_argument("--save_latest_state", action="store_true")

    # model settings
    parser.add_argument("--gpt2_hidden_size", type=int, default=768)
    parser.add_argument("--sum_start_end_emb", action="store_true")
    parser.add_argument("--use_speaker_info", action="store_true")
    parser.add_argument("--use_distance_info", action="store_true")
    parser.add_argument("--distance_groups", action="store_true", default=32)
    parser.add_argument("--metadata_emb_size", type=int, default=20)

    parser.add_argument("--param_init_stdev", type=float, default=0.1)
    parser.add_argument("--rnn_num_layers", type=int, default=1)

    return parser.parse_args()

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

    if args.model_load_path:
        # Load in model and optimizer states
        print("***** Loading model from %s *****" % args.model_load_path)
        ckpt = torch.load(args.model_load_path)
        model.load_state_dict(ckpt["model_state_dict"])

    if not args.unfreeze_gpt2:
        model.freeze_gpt2()
    else:
        print("***** Unfreezing gpt2 parameters *****")
        model.unfreeze_gpt2()

    # load state dict before transferring to GPU to save memory
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    global_step = 0

    if args.model_load_path:
        if not (ckpt["args"].unfreeze_gpt2 ^ args.unfreeze_gpt2)\
            and "optimizer_state_dict" in ckpt:
            print("***** Loading state of optimizer *****")
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        else:
            print("***** Not loading state of optimizer *****")


    num_batches = len(train_dataset)
    # start training
    print("***** Running training *****")
    print("  Num epochs = %d" % args.train_epochs)
    print("  Num batches per epoch = %d" % num_batches)
    print("  Batch size = %d" % args.train_batch_size)
    print("  Logging every %d steps" % args.log_steps)
    if args.eval_and_save_by_epoch:
        print("  Evaluating and saving model every %.2f%% of every epoch" % (args.eval_and_save_by_epoch * 100))

    if args.eval_and_save_by_epoch and args.eval_and_save_by_epoch < 1.0:
        eval_and_save_by_steps = math.floor(num_batches * args.eval_and_save_by_epoch)
    else:
        eval_and_save_by_steps = None

    if args.model_save_path:
        # keep track of best loss for early stopping
        best_loss = float("inf")
        best_step = 0

    total_training_time = 0.0
    epoch = 0
    training_steps_in_this_session = 0
    stepped_in_epoch = None

    if args.model_load_path:
        epoch = ckpt["epoch"]
        global_step = ckpt["global_step"]
        if args.model_save_path and "best_loss" in ckpt:
            best_loss = ckpt["best_loss"]
        # stepped_in_epoch = ckpt["step_in_epoch"]
        # deal with case where one session may not finish one full epoch
        if num_batches == ckpt["num_batches_in_epoch"] \
            and ckpt["step_in_epoch"] != 0 \
            and ckpt["step_in_epoch"] != ckpt["num_batches_in_epoch"] - 1:
            stepped_in_epoch = ckpt["step_in_epoch"]
            print("Fastforwarding to step %d in epoch %d" % (stepped_in_epoch, epoch))

        if ckpt["step_in_epoch"] == ckpt["num_batches_in_epoch"] - 1:
            epoch = ckpt["epoch"] + 1

        # clean up checkpoint to save memory
        del ckpt
        torch.cuda.empty_cache()

    # data to monitor memory usage
    ctx_ids_dim0_sum = 0
    ctx_ids_dim1_sum = 0
    anaphor_ids_dim0_sum = 0
    anaphor_ids_dim1_sum = 0
    start_time = time.time()
    while epoch < args.train_epochs:
        print("*** Epoch %d ***" % epoch)
        for step, batch in enumerate(train_dataloader):
            # fast-forward
            if stepped_in_epoch and step <= stepped_in_epoch:
                continue
            elif stepped_in_epoch:
                stepped_in_epoch = None

            # monitor memory usage
            ctx_ids_dim0_sum += batch["ctx_ids"].shape[0]
            ctx_ids_dim1_sum += batch["ctx_ids"].shape[1]
            anaphor_ids_dim0_sum += batch["anaphor_ids"].shape[0]
            anaphor_ids_dim1_sum += batch["anaphor_ids"].shape[1]

            batch = batch_to_device(batch, device)
            model.zero_grad()
            model.train()

            res_dict = model(batch)
            loss = res_dict["loss"]

            loss.backward()

            optimizer.step()
            global_step += 1
            training_steps_in_this_session += 1

            loss = loss.item()
            del res_dict

            if training_steps_in_this_session < 20 or global_step % args.log_steps == 0:
                avg_time_per_batch = (time.time() - start_time) / training_steps_in_this_session
                estimated_time = (num_batches - (step+1)) * avg_time_per_batch
                print("  step %d/%d, global_step %d, batch loss = %.6f" \
                      % (step+1, num_batches, global_step, loss))
                print("  avg time per batch = %.2f, est %.2f mins left for this epoch" \
                      % (avg_time_per_batch, estimated_time / 60))

            if training_steps_in_this_session < 20 or global_step % (args.log_steps * 5) == 0:
                print("  [tensor dims] ctx_ids [%d, %d], avg [%.2f, %.2f], anaphor_ids [%d, %d], avg [%.2f, %.2f]" \
                      % (batch["ctx_ids"].shape[0], batch["ctx_ids"].shape[1],
                         ctx_ids_dim0_sum / training_steps_in_this_session,
                         ctx_ids_dim1_sum / training_steps_in_this_session,
                         batch["anaphor_ids"].shape[0], batch["anaphor_ids"].shape[1],
                         anaphor_ids_dim0_sum / training_steps_in_this_session,
                         anaphor_ids_dim1_sum / training_steps_in_this_session))

            if eval_and_save_by_steps and step % eval_and_save_by_steps == 0 \
                and step > 0 and step < num_batches - eval_and_save_by_steps:
                best_loss = eval_and_save_checkpoint(args, epoch, eval_dataset,
                    best_loss, step, num_batches, global_step, model,
                    optimizer if args.save_optimizer_state else None,
                    args.save_latest_state)

        best_loss = eval_and_save_checkpoint(args, epoch, eval_dataset,
            best_loss, step, num_batches, global_step, model,
            optimizer if args.save_optimizer_state else None,
            args.save_latest_state)
        epoch += 1

def eval_and_save_checkpoint(args, epoch, eval_dataset, best_loss, step_in_epoch,
                             num_batches_in_epoch, global_step,
                             model, optimizer, save_latest_state):
    eval_results = evaluate(args, model, eval_dataset, global_step)
    eval_loss = eval_results["eval_loss"]
    if args.model_save_path:
        model_checkpoint = {
            "args": args,
            "epoch": epoch,
            "eval_loss": eval_loss,
            "best_loss": min(eval_loss, best_loss),
            "step_in_epoch": step_in_epoch,
            "num_batches_in_epoch": num_batches_in_epoch,
            "global_step": global_step,
            "model_state_dict": model.state_dict()
        }
        if optimizer:
            model_checkpoint["optimizer_state_dict"] = optimizer.state_dict()
        if eval_loss < best_loss:
            best_save_path = args.model_save_path + "_best.ckpt"
            print("  current model has best eval loss, saving to %s" % best_save_path)
            torch.save(model_checkpoint, best_save_path)
        if save_latest_state:
            latest_save_path = args.model_save_path + "_latest.ckpt"
            print("  saving latest version of model to %s" % latest_save_path)
            torch.save(model_checkpoint, latest_save_path)
    best_loss = min(eval_loss, best_loss)
    return best_loss

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
