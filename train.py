import torch

from config import (
    batch_size, block_size, max_iters, eval_interval, learning_rate, device,
    eval_iters, n_embd, n_head, n_layer, n_loop, dropout,
)
from model import GPTLanguageModel, LoopedGPTLanguageModel
# ===== Codex 添加：循环深度逐 token 监测所需的指标函数 =====
from loop_metrics import estimate_loop_metrics, write_metrics_csv
# ===== Codex 添加结束 =====

with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

chars = sorted(list(set(text)))

vocab_size = len(chars)

stoi = { ch:i for i,ch in enumerate(chars) }

encode = lambda s: [stoi[c] for c in s]

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
# ===== Codex 添加：固定验证样本，用于比较训练过程中各循环深度的指标 =====
monitor_batches = [get_batch('val') for _ in range(min(8, eval_iters))]
monitor_history = []
# ===== Codex 添加结束 =====

for iter in range(max_iters):

    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss(looped_model)
        print(f"looped_gpt:step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        # ===== Codex 添加：记录逐轮 loss 分布和 top-1 正确率 =====
        metrics = estimate_loop_metrics(looped_model, monitor_batches)
        monitor_history.extend({'step': iter, **row} for row in metrics)
        print(f"  same-token val: 1 loop loss {metrics[0]['val_loss']:.4f}, "
              f"accuracy {metrics[0]['accuracy']:.2%}; "
              f"{n_loop} loops loss {metrics[-1]['val_loss']:.4f}, "
              f"accuracy {metrics[-1]['accuracy']:.2%}")
        # ===== Codex 添加结束 =====

    xb, yb = get_batch('train')

    logits, loss = looped_model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# ===== Codex 添加：保存训练过程中的逐轮监测结果 =====
write_metrics_csv('loop_train_metrics.csv', monitor_history)
print('Saved loop_train_metrics.csv')
# ===== Codex 添加结束 =====

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

# ==================== 训练结束后的验证指标 ====================
# n_loop 层是训练时的完整循环
# 普通 GPT 也在同一批数据上算一次，作为对照
print("Evaluating looped_gpt with 1..n_loop loops and the normal gpt on validation set...")

eval_batches = [get_batch('val') for _ in range(eval_iters)]
# ===== Codex 添加：在同一批 token 上统计所有循环深度，并保存详细指标 =====
final_metrics = estimate_loop_metrics(looped_model, eval_batches)
write_metrics_csv('loop_token_metrics.csv', final_metrics)
loop_val_losses = {row['n_loops']: row['val_loss'] for row in final_metrics}
# ===== Codex 添加结束 =====
gpt_val_loss = estimate_loss_on_batches(model, eval_batches)

# ===== Codex 添加：输出逐轮正确率与 token 状态变化 =====
for row in final_metrics:
    transition = (f", improved {row['fraction_improved']:.1%}, flat {row['fraction_flat']:.1%}, "
                  f"wrong->right {row['became_correct']:.1%}, right->wrong {row['became_wrong']:.1%}"
                  if row['n_loops'] > 1 else '')
    print(f"looped_gpt:{row['n_loops']} loops: val loss {row['val_loss']:.4f}, "
          f"accuracy {row['accuracy']:.2%}{transition}")
print(f"gpt:full model: val loss {gpt_val_loss:.4f}")
print('Saved loop_token_metrics.csv')
# ===== Codex 添加结束 =====

# 保存原始数值，方便以后复查或重新画图（最后一行为普通 GPT 的对照 loss）
with open('loop_val_losses.csv', 'w') as f:
    f.write('n_loops,val_loss\n')
    for n_loops, val_loss in loop_val_losses.items():
        f.write(f'{n_loops},{val_loss:.6f}\n')
    f.write(f'gpt,{gpt_val_loss:.6f}\n')
