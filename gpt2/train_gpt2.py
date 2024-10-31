import torch
import torch.nn as nn
from torch.nn import functional as F
from dataclasses import dataclass
import tiktoken
import time




@dataclass
class GPTConfig:
    block_size: int = 1024  # max sequence length 
    vocab_size: int = 50257  # number of tokens, 50k BPE merges + 256 byte tokens + 1 end token
    n_layer: int = 12  # number of layers 
    n_head: int = 12  # number of heads
    n_embd: int = 768  # embedding dimension


class DataLoaderLite():
    """
    Class for loading, encoding and decoding text data using gpt2 tokinizer. Samples batches sequentially
    by slicing tokens, starting at the first one and moving back to it as it reaches the last one.
    """

    def __init__(self, B, T) -> None:
        self.B, self.T = B, T
        # state
        self.current_position = 0  
        
        # read the data, tokenize it and save as tensor
        with open('input.txt', 'r', encoding='utf-8') as file:
            text = file.read()
        enc = tiktoken.get_encoding('gpt2')
        tokens = enc.encode(text)
        self.tokens = torch.tensor(tokens)

    def next_batch(self) -> torch.Tensor:
        B, T = self.B, self.T
        buff = self.tokens[self.current_position:self.current_position + B * T + 1]
        x = buff[:-1].view(B, T)
        y = buff[1:].view(B, T)
        # update pointer state
        self.current_position += B * T
        # if loading next batch would be out of bounds, reset
        if self.current_position + (B * T + 1) > len(self.tokens):
            self.current_position = 0
        return x, y
    """
    def decode(self, x: torch.Tensor) -> None:
        assert x.dim() == 2, f"Expected 2-D tensor, but recieved {x.dim()}-D one."
        list_x = x.tolist()
        for i, row in enumerate(list_x):
            decoding = self.tokenizer.decode(row)
            print(f"<{i}>\n{decoding}")
        print("<end>")
    """


class CasualSelfAttention(nn.Module):
    """
    Class for Multi-Head Self-Attention, nothing too crazy.
    """

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()

        assert config.n_embd % config.n_head == 0, "Unable to evenly split the embedding vector across all heads."
        self.head_size = config.n_embd // config.n_head
        self.n_head = config.n_head
        self.n_embd = config.n_embd

        # [[query, key, value], ...] merged in a singe dimension
        self.c_attn = nn.Linear(config.n_embd, 3 * self.n_embd)
        # projection layer applied before residual connection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        # untrained masking tensor, stored as a evvicent torch buffer
        self.register_buffer('bias', torch.tril(torch.ones(config.block_size, config.block_size))
                             .view(1, 1, config.block_size, config.block_size))  # reshape manually, explicity!

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape  # C = n_embd
        qkv = self.c_attn(x)  # (B, T, C), C = 3 * n_embd
        q, k, v = torch.split(qkv, self.n_embd, dim=2)  # (B, T, C), C = n_embd
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        # attention (materializes the large (T,T) matrix for all the queries and keys)
        """
        att = q @ k.transpose(-2, -1) * (k.size(-1)**-0.5)
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))  # slice for sequences shorter than block size
        att = F.softmax(att, dim=-1)
        y = att @ v
        """
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # re-assemble all head outputs side by side
        # output projection
        y = self.c_proj(y)
        return y


class MLP(nn.Module):

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)  # fc - fully connected
        self.gelu = nn.GELU(approximate="tanh")  # same non-linearity was used in gpt2 training
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)  # project before residual connection
        self.c_proj.NANOGPT_SCALE_INIT = 1


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CasualSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor):
        # normalization -> action -> residual connection
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),  # token embeddings
            wpe = nn.Embedding(config.block_size, config.n_embd),  # position embeddings
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),  # stacked blocks
            ln_f = nn.LayerNorm(config.n_embd)  # final normalization
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size)  # language modeling head

        # weight sharing scheme
        self.transformer.wte.weight = self.lm_head.weight
        # follow gpt2 paper initialization
        self.apply(self._init_weights)
    
    def _init_weights(self, module) -> None:
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, "NANOGPT_SCALE_INIT"):
                std *= (2 * self.config.n_layer) ** -0.5  # every block has both attention and mlp added
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor = None) -> list[torch.Tensor, torch.Tensor]:
        B, T = idx.shape
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is {self.config.block_size}."

        # embed tokens and their positions together as x
        token_embds = self.transformer.wte(idx)  # (B, T, n_embd)
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)  # (T)
        pos_embds = self.transformer.wpe(pos)  # (T, n_embd)
        x = token_embds + pos_embds  # (B, T, n_embd) 

        # pass x through blocks and apply final normalization
        for block in self.transformer.h:
            x = block(x)  # (B, T, n_embd)
        
        # turn embeddings to logits of vocabulary size
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))  # cross_entropy() takes flat tansors
        return logits, loss

    def generate(self, inputs: torch.Tensor, num_chars: int) -> None:
        # B, T = inputs.shape

        for i in range(num_chars):
            # cut on the left amount of characters we added on the right (keep T fixed)
            buff = inputs[:, i:]
            # forward the model
            probs = self(buff)  # (B, T, vocab_size)
            # get probabilities for the last character in each example
            probs_last = probs[:, -1, :]  # (B, vocab_size)
            # predict next character for each example based on the probabilities
            next_chars = torch.multinomial(probs_last, num_samples=1)  # (B, 1)
            # add predicted letter to each example
            inputs = torch.cat([inputs, next_chars], dim=1)  # (B, T + i)
        return inputs


model = GPT(GPTConfig())
train_loader = DataLoaderLite(B=4, T=32)

optimizer = torch.optim.AdamW(params=model.parameters(), lr=3e-4)
for i in range(50):
    t0 = time.time()
    # clean gradients
    optimizer.zero_grad(set_to_none=True)
    # load new batch
    Xb, Yb = train_loader.next_batch()
    # forward pass
    logits, loss = model(Xb, Yb)
    # backward pass
    loss.backward()
    # update parameters
    optimizer.step()
    t1 = time.time()
    dt = (t1 - t0) * 1000 # time diff in miliseconds
    tokens_per_sec = (train_loader.B * train_loader.T) / (t1 - t0)
    log = f"step: {i} | loss: {loss.item()} | time: {dt:.2f}ms | tok/sec: {tokens_per_sec:.2f}"
    print(log)
