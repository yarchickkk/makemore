import torch
import torch.nn as nn
from torch.nn import functional as F


# Hyperparameters
batch_size = 32
block_size = 8
max_iters = 5000
eval_interval = 300
learning_rate = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 32
n_layers = 3
dropout = 0.2

# Make the results of random generation constant.
torch.manual_seed(13)

# Read in all the words.
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

chars = sorted(set(text))
vocab_size = len(chars)
stoi = {ch:i for i, ch in enumerate(chars)}
itos = {i:ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data, val_data = data[:n], data[n:]


def get_batch(split: str) -> tuple[torch.tensor, torch.tensor]:
    data = train_data if split == 'train' else val_data
    ix = torch.randint(0, len(data) - block_size, (batch_size, ))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = m(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out 


class Head(nn.Module):

    def __init__(self, head_size: int) -> None:
        super().__init__()
        # Excpect (B, T, C), where C = n_embd
        self.query = nn.Linear(n_embd, head_size, bias=False)  
        self.key = nn.Linear(n_embd,  head_size, bias=False)
        self.value = nn.Linear(n_embd,  head_size, bias=False)
        # Decoder block
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.tensor) -> torch.tensor:
        B, T, C = x.shape
        q, k, v = self.query(x), self.key(x), self.value(x)                  # (B, T, head_size)
        # Get weights
        wei = q @ k.transpose(-1, -2) * C**-0.5                              # (B, T, T)
        wei = torch.masked_fill(wei, self.tril[:T, :T] == 0, float('-inf'))  # (B, T, T)
        wei = F.softmax(wei, dim=1)                                          # (B, T, T)
        # Prevent some nodes from connunicating
        wei = self.dropout(wei)  
        # Aggregate values
        out = wei @ v                                                        # (B, T, head_size)
        return out


class MultiHeadAttention(nn.Module):

    def __init__(self, num_heads: int, head_size: int) -> None:
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.tensor) -> torch.tensor:
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        # Residual connection: project result before adding
        out = self.proj(out)
        out = self.dropout(out)  # Turn off some neurons
        return out


class FeedForward(nn.Module):

    def __init__(self, n_embd: int) -> None:
        super().__init__()
        # MLP
        self.net = nn.Sequential(
            nn.Linear(n_embd, n_embd),
            nn.ReLU(),
            # Residual connection: project result before adding
            nn.Linear(n_embd, n_embd),
            nn.Dropout(dropout)  # Turn off some neurons
        )
    
    def forward(self, x: torch.tensor) -> torch.tensor:
        return self.net(x)


class Block(nn.Module):

    def __init__(self, n_embd: int, num_heads: int) -> None:
        super().__init__()
        # Communication, computation, normalization for both
        head_size = n_embd // num_heads
        self.sa = MultiHeadAttention(num_heads, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x: torch.tensor) -> torch.tensor:
        # Residual connection: add result of computation to the input
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x)) 
        return x


class BigramLanguageModel(nn.Module):
    
    def __init__(self) -> None:
        super().__init__()
        # Each token has it's own identity and position in a batch
        self.identity_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        # Transformer blocks
        self.blocks = nn.Sequential(*[Block(n_embd, num_heads=4) for _ in range(n_layers)])
        # Normaliztion before the last layer
        self.ln_f = nn.LayerNorm(n_embd)
        # Converts embedding vector to one of vocabulary size
        self.lm_head = nn.Linear(n_embd, vocab_size)
        
    def forward(self, idx: torch.tensor, targets: torch.tensor = None) -> torch.tensor:   
        B, T = idx.shape

        # Tokens to transformer's inputs
        identity_embeddings = self.identity_embedding_table(idx)              # (B, T)
        position_embeddings = self.position_embedding_table(torch.arange(T))  #    (T)
        x = identity_embeddings + position_embeddings                         # (B, T)
        # Apply transformer, final normalization, convert to logits.
        x = self.blocks(x)        # (B, T, C), C = n_embd
        x = self.ln_f(x)          # (B, T, C), C = n_embd
        logits = self.lm_head(x)  # (B, T, C), C = vocab_size
    
        if targets is None:
            loss = None
        else:
            # F.cross_entropy expects input of shape (N, C), where N is a batch size
            B, T, C = logits.shape
            logits = logits.view(B * T, C)  # Pass two batch dimension we had as one
            targets = targets.view(B * T)   # Reshape targets accordingly
            loss = F.cross_entropy(logits, targets) if targets is not None else None
        
        return logits, loss
    
    def generate(self, idx: torch.tensor, max_new_tokens: int = 100) -> None:
        # "idx" is tensor of shape (B, T)
        for _ in range(max_new_tokens):
            # Cut "idx" to the length of "block_size" in second dimension
            forward_idx = idx[:, -block_size:] if idx.shape[-1] > block_size else idx
            # Forward pass
            logits, loss = self(forward_idx)                    # (B, T, C)
            logits = logits[:, -1, :]                           # (B, C)
            # Normalize the last dimension
            probs = F.softmax(logits, dim=-1)                   # (B, C)
            # Select single token
            next_idx = torch.multinomial(probs, num_samples=1)  # (B, 1)
            idx = torch.cat([idx, next_idx], dim=1)             # (B, T + 1)
        
        return idx

torch.autograd.set_detect_anomaly(True)
model = BigramLanguageModel()
m = model.to(device)

# Create Torch optimizer
optimizer = torch.optim.AdamW(m.parameters(), lr=learning_rate)

for iter in range(max_iters):
    # print once in a while
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step: {iter} | train loss: {losses['train']:.4f} | val loss: {losses['val']:.4f}")
    # get batch
    xb, yb = get_batch('train')
    # evaluate loss
    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=300)[0].tolist()))
