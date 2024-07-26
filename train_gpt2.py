import math
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
import inspect
import tiktoken
import time

class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        # key, query and value projections for all heads but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # output projection
        self.c_proj = nn.Linear(config.n_embd,  config.n_embd)
        self.c_proj.NANOGPT_SCALE_FLAG = 1
        #regularization
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size)).view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        # B is batch size
        # T is Context Length (Also called as Block Size B)
        # C is embedding dimension (n_embd)
        B, T, C = x.size()

        # QKV dimensions are [B, T, C, C * C * C]
        qkv = self.c_attn(x)
        # q, k and v are [B, T, C, C] after split, dim=2 is the last dimension (of size C * C * C)
        q, k, v = qkv.split(self.n_embd, dim=2)

        # k now will be [B, T, Nh, C//Nh] --> Nh is number of heads
        # This operations basically splits the n_embd into the number of heads equally
        # so that each head is only looking at part of the n_embd. This is basically
        # what multi headed attention is.
        #.transpose will make it [B, Nh, T, C//Nh]
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        # k.transpose(-2, -1) --> will become [B, Nh, C//Nh, T]
        # q is [B, Nh, T, C//Nh]
        # att is [B, Nh, T, T] (after matrix mul)
        #att = (q @ k.transpose(-2, -1)) * (1.0/math.sqrt(k.size(-1)))
        
        # masked_fill will set the upper triangle of the matrix to 'inf'
        # What this means is that during training - only token occuring
        # before are paid attention to and future tokens are ignored.
        # This is called auto-regressive attention.
        #att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))

        # softmax - Not quite sure if I understand this part correctly
        # softmax seems to be getting applied across the heads (where as )
        # I always thought that should be applied across the context length (T)
        #att = F.softmax(att, dim=-1)

        # y would be [B, Nh, T, C//Nh]
        #y = att @ v

        #  Optimization 3 - Flash Attention Optimization - Reduces the memor required for calculating
        # attention dramatically as it fused the 4 line of attention above in one
        # and also computes in online fashion on smaller chunks of memory.
        # Reduced the training loop time by 27%, while maintaing the same accuracy. 
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        # y.transpose would be [B, T, Nh, C//Nh]
        # contiguous.view will concatenate the C//Nh dimension across Nh 
        # making it C so y would be [B, T, C]
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        #y would be [B, T, C]
        y = self.c_proj(y)
        return y


class MLP(nn.Module):

    def __init__(self, config) -> None:
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_FLAG = 1
    
    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x


class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    
    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x



@dataclass
class GPTConfig:
    block_size: int = 1024 #  max sequence length or Content length or block size
    vocab_size: int = 50257 # GPT2 vocab size
    n_layer: int = 12 # GPT2 layers
    n_head: int = 12 # GPT2 heads
    n_embd: int = 768 # GPT2 embedding size


class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd) 
        ))

        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        #weight between the input embedding layer and the output head layer is shared
        # Attentionn is all you need paper mentions this and refer to some other paper
        self.transformer.wte.weight = self.lm_head.weight

        # Initlaization of parameters - inferred from gpt2 code
        self.apply(self._init_weights)
    

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'NANOGPT_SCALE_FLAG'):
                std = (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    
    def forward(self, idx, targets=None):
        #idx is of the shape (B, T)
        B, T = idx.shape
        assert T <= self.config.block_size, f"Can not forward sequence of length {T}, block size"
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device) # shape (T)
        pos_emb = self.transformer.wpe(pos) # shape (T, n_embed (or C))
        tok_emb = self.transformer.wte(idx) # shape is (B, T, n_embed)
        x = tok_emb + pos_emb

        #forward the blocks of the transformer
        for block in self.transformer.h:
            x = block(x)
        
        #forward the last layer norm
        x = self.transformer.ln_f(x)

        logits = self.lm_head(x) # (B, T, vocab_size)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss
    
    def configure_optimizer(self, weight_decay, learning_rate, betas, device):
        # start with all the parameters that requires grad
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}

        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]

        optim_groups = [
            {'params': decay_params, 'weight_decay':weight_decay},
            {'params': nodecay_params, 'weight_decay':0.0}
        ]

        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")
        return optimizer

    # Copied as is from the github repo (https://github.com/karpathy/build-nanogpt/blob/master/train_gpt2.py)
    # This function basically loads GPT2 model weights from hugging face into our GPT class.
    @classmethod
    def from_pretrained(cls, model_type):
        """Loads pretrained GPT-2 model weights from huggingface"""
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model



#-------------------------------------------------------------------------------
device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
# elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
#     device = "mps"
# MPS onmy laptop is 4x slower then CPU.
print(f"using device :{device}")

torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)


#Simple Data Loader Class
class DataLoaderLite:
    def __init__(self, B, T):
    # Load data from the tinyshakesphere data downloaded into input.txt file
        with open("input.txt", 'r') as file:
            text = file.read()
        enc = tiktoken.get_encoding("gpt2")
        self.tokens = torch.tensor(enc.encode(text))
        self.current_position = 0
        self.B = B
        self.T = T
        print(f"DataLoader Init: Num tokens: {len(self.tokens)}")
        print(f"DataLoader Init: Num Batches: {len(self.tokens) // (self.B  * self.T)}")
    
    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position:self.current_position + B * T + 1]
        x = (buf[:-1]).view(B, T)
        y = (buf[1:]).view(B, T)
        self.current_position += (B * T)
        if self.current_position + (B * T + 1) >= len(self.tokens):
            self.current_position = 0
        return x, y

# GPT3 uses 0.5 million tokens, so we should use the same.
# however that will requires a lot many GPUs. so we would instead
# do a series of micro batches and do graident accumulation
total_batch_size = 2**19
print(total_batch_size)
B = 16
T = 1024
assert total_batch_size % (B * T) == 0, "make sure the total batch size is divisible by B * T"
grad_accum_steps = total_batch_size // (B * T)
print(f"Total desired batch size {total_batch_size}")
print(f"==> calculated gradient accumlation steps {grad_accum_steps}")

train_loader = DataLoaderLite(B=B, T=T)

#model = GPT.from_pretrained('gpt2')
# Optimization 4 - nice number roundup for vocabsize
# 50304 can be divided by 2,4,8,16,128 so works better
# with cuda. The extra padded tokens, the model learns to 
# drive their probabilities to zero. Gives 4 to 30% improvement.
model = GPT(GPTConfig(vocab_size=50304))
print('did not crash yay!!!')

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
model = torch.compile(model)
model.to(device)

# Optimization 7 - Learning Rate scheduler - Cosine with warmup
max_lr = 6e-4
min_lr = max_lr * 0.1
warmup_steps = 10
max_steps = 50

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
optimizer = model.configure_optimizer(weight_decay=0.1, learning_rate=6e-4, betas=(0.9, 0.95), device=device)
# Training loop
for step in range(max_steps):
    t1 = time.time()
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

        # Optimization 6 - Clipping Gradients
        loss = loss / grad_accum_steps
        loss_accum += loss.detach()
        loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    # Set the learning rate 
    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    optimizer.step()
    torch.cuda.synchronize()
    t2 = time.time()
    dt = (t2 - t1)
    token_per_second = (train_loader.B * train_loader.T * grad_accum_steps)/dt
    print(f"for step {step:4d} | loss {loss_accum:.6f} | norm {norm:.4f} | time {dt*1000:0.4f} ms | tok/sec {token_per_second}")

import sys
sys.exit(0)

num_return_sequence = 5
max_length = 30

model.eval()

import tiktoken
enc = tiktoken.get_encoding('gpt2')
tokens = enc.encode("Hello, I'm a language model,")
print(tokens)
tokens = torch.tensor(tokens, dtype=torch.long) # (8, )
tokens = tokens.unsqueeze(0).repeat(num_return_sequence, 1) # (5, 8)
print(tokens.shape)
x = tokens
x = tokens.to(device)

# Generation
# Input is a B, T where B = 5, T= 8
torch.manual_seed(42)
# torch.cuda.manaul_seed(42)

while x.size(1) < max_length:

    #forward the model to get the logits
    with torch.no_grad():
        logits = model(x)

        # take the logits at the last col
        logits = logits[:, -1, :] #(B, T, vocab_size) 
     
        # softmax for probabilities
        probs = F.softmax(logits, dim=-1)

        # do top k sampling of top 50 tokesn (hugging face does it that way)
        #topk_probs becomes (5, 50)
        topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)

        #select a token from top k
        # ix is (B, 1) or (5, 1) in this case
        ix = torch.multinomial(topk_probs, 1)
        
        # gather the corresponding indices
        xcol = torch.gather(topk_indices, -1, ix) #(B, 1)

        x = torch.cat((x, xcol), dim=1)

for i in range(num_return_sequence):
    tokens = x[i, :max_length].tolist()
    decoded = enc.decode(tokens)
    print(">", decoded)




