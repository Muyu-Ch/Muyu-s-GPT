import torch

from config import (
    batch_size, block_size, max_iters, eval_interval, learning_rate, device,
    eval_iters, n_embd, n_head, n_layer, n_loop, dropout,
)
from model import GPTLanguageModel, LoopedGPTLanguageModel
from loop_metrics import estimate_loop_metrics, write_metrics_csv

# 生图部分用的 matplotlib
import matplotlib
matplotlib.use('Agg')  # 不弹窗，直接把图保存成文件
import matplotlib.pyplot as plt

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
def estimate_loop_losses(model, loop_counts, batches):
    """分别用不同的循环层数计算平均 loss。

    对每个循环层数都使用同一批数据（同一组 X/Y），
    这样不同层数之间的 loss 差异只来自循环次数，而不是采样噪声。
    """
    model.eval()
    out = {}
    for n_loops in loop_counts:
        losses = torch.zeros(len(batches))
        for k, (X, Y) in enumerate(batches):
            _, loss = model.forward_n_loops(X, Y, n_loops=n_loops)
            losses[k] = loss.item()
        out[n_loops] = losses.mean().item()
    model.train()
    return out

@torch.no_grad()
def estimate_loss_on_batches(model, batches):
    """普通 GPT 在同一批验证数据上的平均 loss，用于和循环模型公平对比。"""
    model.eval()
    losses = torch.zeros(len(batches))
    for k, (X, Y) in enumerate(batches):
        _, loss = model(X, Y)
        losses[k] = loss.item()
    model.train()
    return losses.mean().item()

model = GPTLanguageModel(
    vocab_size=vocab_size,
    n_embd=n_embd,
    n_head=n_head,
    n_layer=n_layer,
    block_size=block_size,
    dropout=dropout,
)
looped_model = LoopedGPTLanguageModel(
    vocab_size=vocab_size,
    n_embd=n_embd,
    n_head=n_head,
    n_loop=n_loop,
    block_size=block_size,
    dropout=dropout,
)
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
monitor_batches = [get_batch('val') for _ in range(min(8, eval_iters))]
monitor_history = []

for iter in range(max_iters):

    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss(looped_model)
        print(f"looped_gpt:step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        metrics = estimate_loop_metrics(looped_model, monitor_batches)
        monitor_history.extend({'step': iter, **row} for row in metrics)
        print(f"  same-token val: 1 loop loss {metrics[0]['val_loss']:.4f}, "
              f"accuracy {metrics[0]['accuracy']:.2%}; "
              f"{n_loop} loops loss {metrics[-1]['val_loss']:.4f}, "
              f"accuracy {metrics[-1]['accuracy']:.2%}")

    xb, yb = get_batch('train')

    logits, loss = looped_model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

write_metrics_csv('loop_train_metrics.csv', monitor_history)
print('Saved loop_train_metrics.csv')

torch.save({
    'model_state_dict': model.state_dict(),
    'config': {
        'n_embd': n_embd,
        'n_head': n_head,
        'n_layer': n_layer,
        'block_size': block_size,
        'dropout': dropout,
    },
    'chars': chars,
}, 'gpt_checkpoint.pth')

torch.save({
    'model_state_dict': looped_model.state_dict(),
    'config': {
        'n_embd': n_embd,
        'n_head': n_head,
        'n_loop': n_loop,
        'block_size': block_size,
        'dropout': dropout,
    },
    'chars': chars,
}, 'looped_gpt_checkpoint.pth')

print("Generating text from gpt model...")

context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))
open('more.txt', 'w').write(decode(m.generate(context, max_new_tokens=10000)[0].tolist()))

print("Generating text from looped_gpt model (full n_loop loops)...")

lmcontext = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(lm.generate(lmcontext, max_new_tokens=500)[0].tolist()))
open('looped_more.txt', 'w').write(decode(lm.generate(lmcontext, max_new_tokens=10000)[0].tolist()))

# ==================== 计算部分：1..n_loop 层验证 loss ====================
# 1..n_loop-1 层只检测 loss，不生成文本；n_loop 层是训练时的完整循环
# 普通 GPT 也在同一批数据上算一次，作为对照
print("Evaluating looped_gpt with 1..n_loop loops and the normal gpt on validation set...")

eval_batches = [get_batch('val') for _ in range(eval_iters)]
loop_counts = list(range(1, n_loop + 1))
final_metrics = estimate_loop_metrics(looped_model, eval_batches)
write_metrics_csv('loop_token_metrics.csv', final_metrics)
loop_val_losses = {row['n_loops']: row['val_loss'] for row in final_metrics}
gpt_val_loss = estimate_loss_on_batches(model, eval_batches)

for row in final_metrics:
    transition = (f", improved {row['fraction_improved']:.1%}, flat {row['fraction_flat']:.1%}, "
                  f"wrong->right {row['became_correct']:.1%}, right->wrong {row['became_wrong']:.1%}"
                  if row['n_loops'] > 1 else '')
    print(f"looped_gpt:{row['n_loops']} loops: val loss {row['val_loss']:.4f}, "
          f"accuracy {row['accuracy']:.2%}{transition}")
print(f"gpt:full model: val loss {gpt_val_loss:.4f}")
print('Saved loop_token_metrics.csv')

# 保存原始数值，方便以后复查或重新画图（最后一行为普通 GPT 的对照 loss）
with open('loop_val_losses.csv', 'w') as f:
    f.write('n_loops,val_loss\n')
    for n_loops, val_loss in loop_val_losses.items():
        f.write(f'{n_loops},{val_loss:.6f}\n')
    f.write(f'gpt,{gpt_val_loss:.6f}\n')

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
    label='Looped GPT (same Block reused)',
)
# 完整循环（训练时的配置）单独高亮
ax.plot(
    counts[-1], losses[-1],
    color='#d64550', linestyle='none',
    marker='o', markersize=10,
    markeredgecolor='#fcfcfb', markeredgewidth=2,
    label=f'Looped GPT, full {n_loop} loops (trained)',
)
# 普通 GPT（不循环）作为水平参考线
ax.axhline(
    gpt_val_loss,
    color='#e08a2e', linewidth=2, linestyle='--',
    label=f'Normal GPT ({n_layer} layers, no looping)',
)

ax.set_xticks(counts)
ax.set_xlim(0.6, n_loop + 0.6)
ax.set_xlabel('Number of loop iterations', fontsize=10, color='#52514e')
ax.set_ylabel('Validation loss', fontsize=10, color='#52514e')
ax.set_title('Looped GPT: validation loss vs. number of loops', fontsize=13,
             color='#0b0b0b', loc='left', pad=26)
ax.text(
    0.0, 1.02,
    f'Trained with {n_loop} loops; both models evaluated on the same validation batches.',
    transform=ax.transAxes, fontsize=9, color='#52514e',
)
ax.text(
    0.98, 0.96,
    f'best: {best_n} loops\nval loss {loop_val_losses[best_n]:.4f}\n'
    f'normal GPT: {gpt_val_loss:.4f}',
    transform=ax.transAxes, fontsize=9, color='#52514e',
    ha='right', va='top',
)
ax.legend(
    loc='center right', fontsize=8.5, frameon=True,
    facecolor='#fcfcfb', edgecolor='#e1e0d9', framealpha=1.0,
    labelcolor='#52514e',
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
for n_loops in [4, 8, 12]:
    print(f"Generating text from less_looped_gpt model ({n_loops} loops)...")

    less_lmcontext = torch.zeros((1, 1), dtype=torch.long, device=device)
    print(decode(lm.generate_n_loops(less_lmcontext, max_new_tokens=500, n_loops=n_loops)[0].tolist()))
    open(f'less_looped_{n_loops}_more.txt', 'w').write(
        decode(lm.generate_n_loops(less_lmcontext, max_new_tokens=10000, n_loops=n_loops)[0].tolist())
    )
