import torch

import torch.nn as nn

from torch.nn import functional as F

# 生图部分用的 matplotlib
import matplotlib
matplotlib.use('Agg')  # 不弹窗，直接把图保存成文件
import matplotlib.pyplot as plt

batch_size = 32

block_size = 24

max_iters = 5000

eval_interval = 500

learning_rate = 3e-4

device = 'cuda' if torch.cuda.is_available() else 'cpu'

eval_iters = 200

n_embd = 128

n_head = 8

n_layer = 6

n_loop=8

dropout = 0.2

torch.manual_seed(1337)

with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

chars = sorted(list(set(text)))

vocab_size = len(chars)

stoi = { ch:i for i,ch in enumerate(chars) }

itos = { i:ch for i,ch in enumerate(chars) }

encode = lambda s: [stoi[c] for c in s]

decode = lambda l: ''.join([itos[i] for i in l])

data = torch.tensor(encode(text), dtype=torch.long)

n = int(0.9*len(data))

train_data = data[:n]

val_data = data[n:]

def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss(model):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

@torch.no_grad()
def estimate_loop_losses(model, loop_counts, split='val'):
    """分别用不同的循环层数计算平均 loss。

    对每个循环层数都使用同一批数据（同一组 X/Y），
    这样不同层数之间的 loss 差异只来自循环次数，而不是采样噪声。
    """
    model.eval()
    batches = [get_batch(split) for _ in range(eval_iters)]
    out = {}
    for n_loops in loop_counts:
        losses = torch.zeros(eval_iters)
        for k, (X, Y) in enumerate(batches):
            _, loss = model.forward_n_loops(X, Y, n_loops=n_loops)
            losses[k] = loss.item()
        out[n_loops] = losses.mean().item()
    model.train()
    return out

class Head(nn.Module):
    """ one head of self-attention 自注意力机制中的单个注意力头 """

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('trill', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5
        wei = wei.masked_fill(self.trill[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        out = wei @ v
        return out

class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel 并行运行的多个自注意力头 """

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity 一个简单的线性层，后接非线性激活函数 """

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """ Transformer block: communication followed by computation Transformer 块：先通信（注意力），再计算（前馈网络） """

    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x+self.sa(self.ln1(x))
        x = x+self.ffwd(self.ln2(x))
        return x

class GPTLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(
            *[Block(n_embd, n_head=n_head) for _ in range(n_layer)]
        )
        # self.blocks=Block(n_embd, n_head=n_head)
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        x = tok_emb + pos_emb
        x=self.blocks(x)
        # for i in range(n_layer):
        #     x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits,loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


class LoopedGPTLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        # self.blocks = nn.Sequential(
        #     *[Block(n_embd, n_head=n_head) for _ in range(n_layer)]
        # )
        self.blocks=Block(n_embd, n_head=n_head)
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        # 默认走训练时监督的完整 n_loop 次循环
        return self.forward_n_loops(idx, targets, n_loops=n_loop)

    def forward_n_loops(self, idx, targets=None, n_loops=n_loop):
        """把同一个 Block 重复作用 n_loops 次后再接 lm_head。

        n_loops 可以小于训练时的 n_loop，用来检验浅层循环是否也具备预测能力。
        """
        B, T = idx.shape

        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        x = tok_emb + pos_emb
        #x=self.blocks(x)
        for i in range(n_loops):
            x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits,loss


    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

    @torch.no_grad()
    def generate_n_loops(self, idx, max_new_tokens, n_loops=n_loop):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, loss = self.forward_n_loops(idx_cond, n_loops=n_loops)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

model = GPTLanguageModel()
looped_model = LoopedGPTLanguageModel()
m = model.to(device)
print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')

lm=looped_model.to(device)
print(sum(p.numel() for p in lm.parameters())/1e6, 'LM parameters')

print("Training gpt model...")

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):

    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss(model)
        print(f"gpt:step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    xb, yb = get_batch('train')

    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

print("Training looped_gpt model...")

optimizer = torch.optim.AdamW(looped_model.parameters(), lr=learning_rate)

for iter in range(max_iters):

    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss(looped_model)
        print(f"looped_gpt:step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    xb, yb = get_batch('train')

    logits, loss = looped_model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

print("Generating text from gpt model...")

context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))
open('more.txt', 'w').write(decode(m.generate(context, max_new_tokens=10000)[0].tolist()))

print("Generating text from looped_gpt model (full n_loop loops)...")

lmcontext = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(lm.generate(lmcontext, max_new_tokens=500)[0].tolist()))
open('looped_more.txt', 'w').write(decode(lm.generate(lmcontext, max_new_tokens=10000)[0].tolist()))

# ==================== 计算部分：1..n_loop-1 层验证 loss ====================
# 1..n_loop-1 层只检测 loss，不生成文本
print("Evaluating looped_gpt with 1..n_loop-1 loops on validation set...")

loop_counts = list(range(1, n_loop))
loop_val_losses = estimate_loop_losses(looped_model, loop_counts, split='val')

for n_loops, val_loss in loop_val_losses.items():
    print(f"looped_gpt:{n_loops} loops: val loss {val_loss:.4f}")

# 保存原始数值，方便以后复查或重新画图
with open('loop_val_losses.csv', 'w') as f:
    f.write('n_loops,val_loss\n')
    for n_loops, val_loss in loop_val_losses.items():
        f.write(f'{n_loops},{val_loss:.6f}\n')

# ==================== 计算部分结束 ====================

# ==================== 生图部分：验证 loss 随循环层数变化的折线图 ====================
counts = list(loop_val_losses.keys())
losses = [loop_val_losses[n] for n in counts]
best_n = min(counts, key=lambda n: loop_val_losses[n])

fig, ax = plt.subplots(figsize=(8, 5))
fig.patch.set_facecolor('#fcfcfb')
ax.set_facecolor('#fcfcfb')

ax.plot(
    counts, losses,
    color='#2a78d6', linewidth=2,
    marker='o', markersize=8,
    markeredgecolor='#fcfcfb', markeredgewidth=2,
)

ax.set_xticks(counts)
ax.set_xlim(0.6, n_loop - 0.6)
ax.set_xlabel('Number of loop iterations', fontsize=10, color='#52514e')
ax.set_ylabel('Validation loss', fontsize=10, color='#52514e')
ax.set_title('Looped GPT: validation loss vs. number of loops', fontsize=13,
             color='#0b0b0b', loc='left', pad=26)
ax.text(
    0.0, 1.02,
    f'Trained with {n_loop} loops; 1-{n_loop - 1} evaluated on the same validation batches.',
    transform=ax.transAxes, fontsize=9, color='#52514e',
)
ax.text(
    0.98, 0.96,
    f'best: {best_n} loops\nval loss {loop_val_losses[best_n]:.4f}',
    transform=ax.transAxes, fontsize=9, color='#52514e',
    ha='right', va='top',
)

ax.grid(axis='y', color='#e1e0d9', linewidth=0.8)
ax.set_axisbelow(True)
for spine in ('top', 'right'):
    ax.spines[spine].set_visible(False)
for spine in ('left', 'bottom'):
    ax.spines[spine].set_color('#c3c2b7')
ax.tick_params(colors='#898781', labelsize=9)

fig.tight_layout()
fig.savefig('looped_val_loss_vs_loops.png', facecolor=fig.get_facecolor())
plt.close(fig)
print("Saved loop loss plot to looped_val_loss_vs_loops.png")
# ==================== 生图部分结束 ====================

# 用 2、4、6 层循环生成文本，输出格式与原来的 4 层版本一致
for n_loops in [2, 4, 6]:
    print(f"Generating text from less_looped_gpt model ({n_loops} loops)...")

    less_lmcontext = torch.zeros((1, 1), dtype=torch.long, device=device)
    print(decode(lm.generate_n_loops(less_lmcontext, max_new_tokens=500, n_loops=n_loops)[0].tolist()))
    open(f'less_looped_{n_loops}_more.txt', 'w').write(
        decode(lm.generate_n_loops(less_lmcontext, max_new_tokens=10000, n_loops=n_loops)[0].tolist())
    )