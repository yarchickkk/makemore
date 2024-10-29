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
    """
    Class for loading, encoding and decoding text data using gpt2 tokinizer. Samples batches sequentially
    by slicing tokens, starting at the first one and moving back to it as it reaches the last one.
    """

    def __init__(self, config: GPTConfig) -> None:
        self.bs = config.block_size
        self.n = config.n_examples
        self.ptr = 0  # initialize pointer
        self.tokenizer = tiktoken.get_encoding('gpt2')
        
        # read the data, tokenize it and save as tensor
        with open('input.txt', 'r', encoding='utf-8') as file:
            text = file.read()
        self.text = torch.tensor(self.tokenizer.encode(text))
        self.text_length = self.text.shape[0]

    def next_batch(self) -> torch.Tensor:
        # check wether there's enough elements left for a new batch
        if self.ptr + self.bs * self.n < self.text_length:
            pass
        else:
            self.ptr = 0  # if not, go to the first token
        
        end = self.ptr + self.bs * self.n  # end of the batch
        X = self.text[self.ptr:end]  # inputs
        Y = self.text[self.ptr + 1:end + 1]  # targets, shifted by 1 to get next characters
        
        self.ptr += 1
        return tuple(s.view(self.n, self.bs) for s in (X, Y))  # 1D -> 2D

    def decode(self, x: torch.Tensor) -> None:
        assert x.dim() == 2, f"Expected 2-D tensor, but recieved {x.dim()}-D one."
        list_x = x.tolist()
        for i, row in enumerate(list_x):
            decoding = self.tokenizer.decode(row)
            print(f"<{i}>\n{decoding}")
        print("<end>")

class MultiHeadSelfAttention(nn.Module):
    """
    Class for Multi-Head Self-Attention, nothing too crazy.
    """

    def __init__(self, config: GPTConfig) -> None:  # COMPARE!
        super().__init__()
        self.config = config

        assert self.config.n_embd % self.config.n_head == 0, "Unable to evenly split the embedding vector across all heads."
        self.head_size = self.config.n_embd // self.config.n_head

        # [[query, key, value], ...] merged in a singe dimension
        self.qkv = nn.Linear(self.config.n_embd, 3 * self.head_size)
        # projection layer applied before residual connection
        self.proj = nn.Linear(self.config.n_embd, self.config.n_embd)
        # untrained masking tensor, stored as a evvicent torch buffer
        self.register_buffer('tril', torch.tril(torch.ones(self.config.block_size, self.config.block_size)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape  # C = n_embd
        H = self.config.n_head

        # add head dimension
        x = torch.unsqueeze(x, 1).repeat(1, H, 1, 1)  # (B, H, T, C), C = n_embd
        # turn each embedding to [q, k, v]
        merged_qkv = self.qkv(x)
        # split on separate q, k, v tensors
        q, k, v = merged_qkv.split(self.head_size, dim=3)  # (B, H, T, C), C = head_size

        # get weights multiplying querys and keys and mask them
        wei = q @ k.transpose(2, 3)  # (B, H, T, T)
        wei = torch.masked_fill(wei[:, :, :T, :T], self.tril[:T, :T] == 0, float('-inf'))  # slice to sample safely
        wei = torch.exp(wei)
        # obtain weighted aggregation of values
        x = wei @ v  # (B, H, T, C), C = head_size

        # some tricky matrix manipulation to merge heads in embedding
        x = x.permute(0, 2, 1, 3).reshape(B, T, H * self.head_size)  # (B, T, n_heads * head_size | n_embd)
        x = self.proj(x)  # pre-residual projection
        return x


class MLP(nn.Module):

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.config = config

        self.fc = nn.Linear(self.config.n_embd, self.config.n_embd)  # fc - fully connected
        self.gelu = nn.GELU(approximate="tanh")  # same non-linearity was used in gpt2 training
        # projection layer applied before residual connection
        self.proj = nn.Linear(self.config.n_embd, self.config.n_embd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc(x)
        x = self.gelu(x)
        x = self.proj(x)
        return x


class Block(nn.Module):
    
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.config = config

        self.ln_1 = nn.LayerNorm(self.config.n_embd, self.config.n_embd)
        self.attn = MultiHeadSelfAttention(self.config)
        self.ln_2 = nn.LayerNorm(self.config.n_embd, self.config.n_embd)
        self.mlp = MLP(self.config)  # COMPARE!

    def forward(self, x: torch.Tensor):
        # normalization -> action -> residual connection
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.config = config

        self.token_embd_table = nn.Embedding(self.config.vocab_size, self.config.n_embd)
        self.pos_embd_table = nn.Embedding(self.config.block_size, self.config.n_embd)

        # stack multiple talk-think blocks sequantially
        self.layers = nn.Sequential(*[Block(self.config) for _ in range(self.config.n_layer)])

        # final layer normalization and language modeling head
        self.ln_f = nn.LayerNorm(self.config.n_embd, self.config.n_embd)
        self.lm_head = nn.Linear(self.config.n_embd, self.config.vocab_size)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # B, T = inputs.shape

        # embed tokens and their positions as x
        token_embds = self.token_embd_table(inputs)  # (B, T, n_embd)
        pos_embds = self.pos_embd_table(torch.arange(self.config.block_size))  # (T, n_embd)
        x = token_embds + pos_embds  # (B, T, n_embd) 

        # pass x through blocks and apply final normalization
        x = self.layers(x)  # (B, T, n_embd)
        x = self.ln_f(x)  # (B, T, n_embd)
        
        # turn embeddings to logits of vocabulary size
        logits = self.lm_head(x)  # (B, T, vocab_size)
        probs = F.softmax(logits, dim=2)  # (B, T, vocab_size)
        return probs

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
test_model(Xb)


# Generate
res = test_model.generate(Xb, 10)
data.decode(res)
