import torch
import torch.nn as nn
from torch.nn import functional as F
from dataclasses import dataclass
import tiktoken




@dataclass
class GPTConfig:
    block_size: int = 256
    vocab_size: int = 65
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
    n_examples: int = 32


class DataLoader():

    def __init__(self, config: GPTConfig) -> None:
        self.bs, self.n, self.ptr = config.block_size, config.n_examples, 0
        self.tokenizer = tiktoken.get_encoding('gpt2')
        with open('input.txt', 'r', encoding='utf-8') as f:
            text = f.read()
        self.text = torch.tensor(self.tokenizer.encode(text))

    def next_batch(self) -> torch.tensor:
        # keep pointer in bounds
        self.ptr = self.ptr if (self.ptr + self.bs * self.n < self.text.shape[0]) else 0 
        end = self.ptr + self.bs * self.n
        X = self.text[self.ptr:end]
        Y = self.text[self.ptr + 1:end + 1]
        self.ptr += 1
        return tuple(s.view(self.n, self.bs) for s in (X, Y))  # 1D -> 2D
    
    def decode(self) -> None:
        print("Still under development!")  # COMPLETE!
    

class MultiHeadSelfAttention(nn.Module):
    
    def __init__(self, config: GPTConfig) -> None:  # COMPARE!
        super().__init__()
        self.config = config
        self.head_size = self.config.n_embd // self.config.n_head

        self.qkv = nn.Linear(self.config.n_embd, 3 * self.head_size)
        self.proj = nn.Linear(self.config.n_embd, self.config.n_embd)

        self.register_buffer('tril', torch.tril(torch.ones(self.config.block_size, self.config.block_size)))

    def forward(self, ids: torch.tensor) -> torch.tensor:
        B, T, C = ids.shape  # C = n_embd
        H = self.config.n_head

        ids = torch.unsqueeze(ids, 1).repeat(1, H, 1, 1)  # (B, H, T, C), C = n_embd
        q, k, v = self.qkv(ids).split(self.head_size, dim=3)  # (B, H, T, C), C = head_size
        
        wei = q @ k.transpose(2, 3)  # (B, H, T, T)
        wei = torch.masked_fill(wei[:, :, :T, :T], self.tril[:T, :T] == 0, float('-inf'))  # slice to sample safely

        ids = wei @ v  # (B, H, T, C), C = head_size
        ids = ids.permute(0, 2, 1, 3).reshape(B, T, H * self.head_size)  # (B, T, n_heads * head_size | n_embd)
        ids = self.proj(ids)  # pre-residual projection
        return ids
    

class MLP(nn.Module):

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.config = config

        self.fc = nn.Linear(self.config.n_embd, self.config.n_embd)
        self.proj = nn.Linear(self.config.n_embd, self.config.n_embd)

    def forward(self, ids: torch.tensor) -> torch.tensor:
        ids = self.fc(ids)
        ids = F.gelu(ids, approximate='tanh')
        ids = self.proj(ids)
        return ids


class Block(nn.Module):
    
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.config = config

        self.ln_1 = nn.LayerNorm(self.config.n_embd, self.config.n_embd)
        self.attn = MultiHeadSelfAttention(self.config)
        self.ln_2 = nn.LayerNorm(self.config.n_embd, self.config.n_embd)
        self.mlp = MLP(self.config)  # COMPARE!

    def forward(self, ids: torch.tensor):
        ids = ids + self.attn(self.ln_1(ids))
        ids = ids + self.mlp(self.ln_2(ids))
        return ids


class GPT(nn.Module):

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.config = config

        self.token_embd_table = nn.Embedding(self.config.vocab_size, self.config.n_embd)
        self.pos_embd_table = nn.Embedding(self.config.block_size, self.config.n_embd)

        self.layers = nn.Sequential(*[Block(self.config) for _ in range(self.config.n_layer)])

        self.ln_f = nn.LayerNorm(self.config.n_embd, self.config.n_embd)
        self.lm_head = nn.Linear(self.config.n_embd, self.config.vocab_size)

    def forward(self, ids: torch.tensor, targets: torch.tensor):
        B, T = ids.shape

        token_embds = self.token_embd_table(ids)  # (B, T, n_embd)
        pos_embds = self.pos_embd_table(torch.arange(self.config.block_size))  # (T, n_embd)
        x = token_embds + pos_embds  # (B, T, n_embd) 
        x = self.layers(x)  # (B, T, n_embd)
        x = self.ln_f(x)  # (B, T, n_embd)
        logits = self.lm_head(x)  # (B, T, vocab_size)
        # Here I need cross Entropy, which requires targets
        # Targets you usually get from a batch
        # Batch you get from a set of data
        # I didn't load it
        print('Got logits!')
    def generate(self) -> None:
        pass
        

test_cfg = GPTConfig(
    block_size = 6,
    vocab_size = 50257,
    n_layer = 3,
    n_head = 4,
    n_embd = 20,
    n_examples = 2  
)
data = DataLoader(test_cfg)
Xb, Yb = data.next_batch()
test_model = GPT(test_cfg)
test_model(Xb, Yb)


# Self attention consists of blocks, where each block is a residual branch:
# LayerNorm -> Masked Multi Head Self-Attention -> LayerNorm -> Feed Forward

# GPT module itself takes care of:
#   token embeddings,               --- 
#   position embeddings,            ---  
#   Attention (separate block)
#   LayerNorm (after blocks),       ---
#   Linear (embeddings -> logits),  ---
#   Softmax (no optimized parameters)

