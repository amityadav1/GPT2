import math
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
import inspect
import tiktoken
import time
import os
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from dataloaderlite import DataLoaderLite
from model import GPT, GPTConfig


#-------------------------------------------------------------------------------

# Optimization 9 - Distributed data parallel
from torch.distributed import init_process_group, destroy_process_group

#Setup DDP, torchrun command sets the appropriate variables
ddp = int(os.environ.get('RANK', -1)) != -1
if ddp:
    assert torch.cuda.is_available(), "Cuda is required for Distributed data parallel"
    init_process_group(backend='nccl')
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0
else:
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    master_process = True
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"

if master_process:
    print(f"using device :{device}")

torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)


# Optimization 8 - Large Batch size 
# GPT3 uses 0.5 million tokens, so we should use the same, 
# This would increase the throughput, although the training time for
# each epoch will increase.
# however that will requires a lot many GPUs. so we would instead
# do a series of micro batches and do graident accumulation
total_batch_size = 2**19
if master_process:
    print(total_batch_size)
B = 16
T = 1024
assert total_batch_size % (B * T * ddp_world_size) == 0, "make sure the total batch size is divisible by B * T * ddp_world_size"
grad_accum_steps = total_batch_size // (B * T * ddp_world_size)
if master_process:
    print(f"Total desired batch size {total_batch_size}")
    print(f"==> calculated gradient accumlation steps {grad_accum_steps}")

train_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_local_rank, num_processes=ddp_world_size, split='train', datadir="edu_fineweb10B", master_process=master_process)
val_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_local_rank, num_processes=ddp_world_size, split='val', datadir="edu_fineweb10B", master_process=master_process)
torch.set_float32_matmul_precision('high')

#model = GPT.from_pretrained('gpt2')
# Optimization 4 - nice number roundup for vocabsize
# 50304 can be divided by 2,4,8,16,128 so works better
# with cuda. The extra padded tokens, the model learns to 
# drive their probabilities to zero. Gives 4 to 30% improvement.
model = GPT(GPTConfig(vocab_size=50304))


# Optimization 2 - Use torch.compile - Torch compile compiles
# the model and eliminate python overhead and also optimized 
# GPU read/writes. The compile increase the compilation time but
# reduces the training time (expected 50% reduction). Torch compile
# can see the entire DAG of the model and then can optimize it so that
# operations can be fused together (kernel fusion). For example if there 
# are 4 different operations happening on an input (elemnetwise raise, then 
# scaler multiplication, then addition back into input etc), without torch.compile
# the python interpreter will run each of them in sequence which would
# copy input and intermediate outputs multiple times between memory and GPU
# thread memory and registers. With torch.compile that memory is only copied
# once and then all the operations are fused together. 
#model = torch.compile(model)
model.to(device)
raw_model = model
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])

# Optimization 7 - Learning Rate scheduler - Cosine with warmup
max_lr = 6e-4
min_lr = max_lr * 0.1

# The finwebedu dataset as 10B unique tokens and we are processing 0.5M token per steps.
# so for 10B token we would need 10B/0.5M ~ 19073 steps
warmup_steps = 715 # from GPT3 paper warmp up until 315 million token,
max_steps = 19073

def get_lr(it):
    if it < warmup_steps:
        return max_lr * (it + 1)/warmup_steps
    if it > max_steps:
        return min_lr
    decay_ratio = (it - warmup_steps)/(max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)



# Optimization 5 - Hyperparameter based on the GPT3 paper (since GPT2 paper
# does not have these details). Set the betas and epsilons
# optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), eps=1e-8)
optimizer = raw_model.configure_optimizer(weight_decay=0.1, learning_rate=6e-4, betas=(0.9, 0.95), device=device, master_process=master_process)

log_dir = "log"
os.mkdir(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"logs.txt")
with open(log_file, 'w') as file:
    pass

# Training loop
for step in range(max_steps):
    t1 = time.time()
    last_step = (step == max_steps - 1)

    # Evaluation loop
    if (step % 250 == 0 or last_step):
        val_loader.reset()
        model.eval()
        with torch.no_grad():
            val_loss_accum = 0.0
            val_loss_step = 20
            for _ in range(val_loss_step):
                x, y = val_loader.next_batch()
                x, y = x.to(device), y.to(device)
                with torch.autocast(device_type=device, dtype=torch.bfloat16):
                    logits, loss = model(x, y)
                loss = loss / val_loss_step
                val_loss_accum += loss.detach()
            if ddp:
                dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)
            if master_process:
                print(f"validation loss {val_loss_accum.item():0.4f}")
    

    # Evaluation loop using helloswag
    if (step % 250 == 0 or last_step) and (not use_compile):
        num_correct_norm = 0
        num_total = 0
        for i, example in enumerate(iterate_examples("val")):
            # only process examples where i % ddp_world_size == ddp_rank
            if i % ddp_world_size != ddp_rank:
                continue
            # render the example into tokens and labels
            _, tokens, mask, label = render_example(example)
            tokens = tokens.to(device)
            mask = mask.to(device)
            # get the logits
            with torch.no_grad():
                with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                    logits, loss = model(tokens)
                pred_norm = get_most_likely_row(tokens, mask, logits)
            num_total += 1
            num_correct_norm += int(pred_norm == label)
        # reduce the stats across all processes
        if ddp:
            num_total = torch.tensor(num_total, dtype=torch.long, device=device)
            num_correct_norm = torch.tensor(num_correct_norm, dtype=torch.long, device=device)
            dist.all_reduce(num_total, op=dist.ReduceOp.SUM)
            dist.all_reduce(num_correct_norm, op=dist.ReduceOp.SUM)
            num_total = num_total.item()
            num_correct_norm = num_correct_norm.item()
        acc_norm = num_correct_norm / num_total
        if master_process:
            print(f"HellaSwag accuracy: {num_correct_norm}/{num_total}={acc_norm:.4f}")
            with open(log_file, "a") as f:
                f.write(f"{step} hella {acc_norm:.4f}\n")

    # Inference Sampling
    if step % 100 == 0:
        model.eval()
        num_return_sequence = 4
        max_length = 32
        enc = tiktoken.get_encoding('gpt2')
        tokens = enc.encode("Hello, I'm a language model,")
        tokens = torch.tensor(tokens, dtype=torch.long) # (8, )
        tokens = tokens.unsqueeze(0).repeat(num_return_sequence, 1) # (5, 8)
        xgen = tokens.to(device)
        sample_rng = torch.Generator(device=device)
        sample_rng.manual_seed(42 + ddp_rank)

        while xgen.size(1) < max_length:

            #forward the model to get the logits
            with torch.no_grad():
                with torch.autocast(device_type=device, dtype=torch.bfloat16):
                    logits, loss = model(xgen)

                # take the logits at the last col
                logits = logits[:, -1, :] #(B, T, vocab_size) 
            
                # softmax for probabilities
                probs = F.softmax(logits, dim=-1)

                # do top k sampling of top 50 tokesn (hugging face does it that way)
                #topk_probs becomes (5, 50)
                topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)

                #select a token from top k
                ix = torch.multinomial(topk_probs, 1, generator=sample_rng)
                
                # gather the corresponding indices
                xcol = torch.gather(topk_indices, -1, ix) #(B, 1)
                xgen = torch.cat((xgen, xcol), dim=1)

        for i in range(num_return_sequence):
            tokens = xgen[i, :max_length].tolist()
            decoded = enc.decode(tokens)
            print(">", decoded)


    #Training Loop
    model.train()
    optimizer.zero_grad()
    loss_accum = 0.0
    for micro_step in range(grad_accum_steps):
        x , y = train_loader.next_batch()
        x = x.to(device)
        y = y.to(device)

        # Optimization 1 - Mixed precision training
        # use bfloat16 instead of float32 for activations etc
        # Autocast automatically decides which parameters to use
        # float16 for (mostly for linear layer) and which one to keep
        # float32 (gradient accumulation etc)
        # this makes operations lile mat mul faster. The expected
        # improved is about 3 to 4x reduction in the training time and
        # corresponding improvement in the token per second throughput.
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            logits, loss = model(x, y)

       
        loss = loss / grad_accum_steps
        loss_accum += loss.detach()
        # With DDP the gradients are averaged across all GPU nodes.
        # However, since we are using micro_batches, the gradient average
        # (using all reduce collective communication) should only be done once 
        # all the micro_batches are processed. The following flag is used to disable
        # synchronization primitives between GPUs, so we only enable the flag once we 
        # are at the last micro_step.
        if ddp:
            model.required_backward_grad_sync = (micro_step == grad_accum_steps - 1)
        loss.backward()
    if ddp:
        dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)    
     # Optimization 6 - Clipping Gradients
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    # Set the learning rate 
    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    optimizer.step()
    torch.cuda.synchronize()
    t2 = time.time()
    dt = (t2 - t1)
    token_per_second = (train_loader.B * train_loader.T * grad_accum_steps * ddp_world_size)/dt
    if master_process:
        print(f"for step {step:4d} | loss {loss_accum:.6f} | norm {norm:.4f} | time {dt*1000:0.4f} ms | tok/sec {token_per_second:0.4f}")


if ddp:
    destroy_process_group()
import sys
sys.exit(0)






