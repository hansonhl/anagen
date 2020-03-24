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
    parser.add_argument("--max_num_ctxs_in_batch", type=int, default=8)
    parser.add_argument("--max_segment_len", type=int, default=512)

    # where to save model
    parser.add_argument("--model_save_path", type=str)

    # gpt2 model settings
    parser.add_argument("--gpt2_model_dir", type=str, default=None)

    # training settings
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--random_seed", type=int, default=39393)
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
    model.freeze_gpt2()
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    num_batches = len(train_dataset)
    # start training
    print("***** Running training *****")
    print("  Num examples = %d" % num_batches)
    print("  Num Epochs = %d" % args.train_epochs)
    print("  Batch size = %d" % args.train_batch_size)
    print("  Logging every %d steps" % args.log_steps)
    print("  Evaluating and saving model every %d steps" % args.eval_and_save_steps)

    # TODO: add checkpoint loading functionality

    global_step = 0
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
            optimizer.step()
            total_training_time += time.time() - start_time
            global_step += 1

            if global_step % args.log_steps == 0:
                avg_time_per_batch = total_training_time / global_step
                estimated_time = (num_batches - (step+1)) * avg_time_per_batch
                print("  step %d/%d, global_step %d, batch loss = %.6f" \
                      % (step+1, num_batches, global_step, loss))
                print("  avg time per batch = %.2f, est remaining time = %.2f mins" \
                      % (avg_time_per_batch, estimated_time / 60))

            if global_step % args.eval_and_save_steps == 0:
                eval_results = evaluate(args, model, eval_dataset)
                # TODO: add tensorboard writer functionality

                if args.model_save_path:
                    print("  saving model to %s" % args.model_save_path)
                    save_path = args.model_save_path + ("_step_%d.ckpt" % global_step)
                    model_checkpoint = {
                        "args": args,
                        "epoch": epoch,
                        "step_in_epoch": step,
                        "global_step": global_step,
                        "model_state_dict": model.state_dict(), # just save everything for now
                        "optimizer_state_dict": optimizer.state_dict()
                    }
                    torch.save(model_checkpoint, args.model_save_path)



def evaluate(args, model, eval_dataset):
    device = torch.device("cuda" if args.gpu else "cpu")
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler,
                                 batch_size=DEFAULT_TRAIN_BATCH_SIZE,
                                 collate_fn=collate)

    print("***** Running evaluation *****")
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

    eval_loss = eval_loss / num_toks
    perplexity = torch.exp(torch.tensor(eval_loss))

    print("***** Eval results *****")
    print("  eval_loss = %.6f" % eval_loss)
    print("  perplexity = %.6f" % perplexity)
