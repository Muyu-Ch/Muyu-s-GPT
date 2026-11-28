import csv
import math

import torch

from config import (
    batch_size,
    block_size,
    max_iters,
    eval_interval,
    learning_rate,
    min_lr,
    warmup_iters,
    weight_decay,
    grad_clip,
    early_stop_patience,
    device,
    eval_iters,
    n_embd,
    n_head,
    n_layer,
    n_loop,
    dropout,
)
from model import GPTLanguageModel, LoopedGPTLanguageModel

# ===== Codex 添加：循环深度逐 token 监测所需的指标函数 =====
from loop_metrics import estimate_loop_metrics, write_metrics_csv
# ===== Codex 添加结束 =====

with open("input.txt", "r", encoding="utf-8") as f:
    text = f.read()

chars = sorted(list(set(text)))

vocab_size = len(chars)

stoi = {ch: i for i, ch in enumerate(chars)}

encode = lambda s: [stoi[c] for c in s]

data = torch.tensor(encode(text), dtype=torch.long)

n = int(0.9 * len(data))

train_data = data[:n]

val_data = data[n:]


def get_batch(split):
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


@torch.no_grad()
def estimate_loss_on_batches(model, batches):
    """在固定的一批数据上估计 loss；GPT 和 looped 模型用同一批 token。"""
    model.eval()
    losses = torch.zeros(len(batches))
    for k, (X, Y) in enumerate(batches):
        _, loss = model(X, Y)
        losses[k] = loss.item()
    model.train()
    return losses.mean().item()


def get_lr(iter):
    """线性 warmup + cosine 衰减；用于控制训练后期的过拟合。"""
    if iter < warmup_iters:
        return learning_rate * (iter + 1) / max(1, warmup_iters)
    progress = (iter - warmup_iters) / max(1, max_iters - warmup_iters)
    return min_lr + 0.5 * (learning_rate - min_lr) * (1 + math.cos(math.pi * progress))


def train_model(model, name, eval_callback=None):
    """训练一个模型：warmup/cosine、weight decay、梯度裁剪、早停 + 最优 checkpoint。"""
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        betas=(0.9, 0.95),
        weight_decay=weight_decay,
    )
    best_val = float("inf")
    best_step = -1
    best_state = None
    bad_evals = 0
    history = []

    for iter in range(max_iters):
        lr = get_lr(iter)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        if iter % eval_interval == 0 or iter == max_iters - 1:
            train_loss = estimate_loss_on_batches(model, train_eval_batches)
            val_loss = estimate_loss_on_batches(model, val_eval_batches)
            history.append(
                {
                    "model": name,
                    "step": iter,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "lr": lr,
                }
            )
            print(
                f"{name}:step {iter}: train loss {train_loss:.4f}, "
                f"val loss {val_loss:.4f}, lr {lr:.2e}"
            )

            if val_loss < best_val:
                best_val = val_loss
                best_step = iter
                best_state = {
                    k: v.detach().cpu().clone() for k, v in model.state_dict().items()
                }
                bad_evals = 0
            else:
                bad_evals += 1

            if eval_callback is not None:
                eval_callback(model, iter)

            if bad_evals >= early_stop_patience:
                print(
                    f"{name}: early stop at step {iter}; "
                    f"best val {best_val:.4f} @ step {best_step}"
                )
                break

        xb, yb = get_batch("train")

        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

    # 最后保存和评估都用验证 loss 最好的那个 checkpoint，而不是过拟合后的最后一步
    if best_state is not None:
        model.load_state_dict(best_state)
        print(
            f"{name}: restored best checkpoint from step {best_step} "
            f"(val loss {best_val:.4f})"
        )

    return history, best_step, best_val


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
print(sum(p.numel() for p in m.parameters()) / 1e6, "M parameters")

lm = looped_model.to(device)
print(sum(p.numel() for p in lm.parameters()) / 1e6, "LM parameters")

# 固定的评估集：两个模型、每个 checkpoint 都在完全相同的 token 上比较；
# 验证集不参与训练，早停和最优 checkpoint 的选择都只看它。
train_eval_batches = [get_batch("train") for _ in range(min(50, eval_iters))]
val_eval_batches = [get_batch("val") for _ in range(eval_iters)]

print("Training gpt model...")

gpt_history, gpt_best_step, gpt_best_val = train_model(model, "gpt")

print("Training looped_gpt model...")

# ===== Codex 添加：固定验证样本，用于比较训练过程中各循环深度的指标 =====
monitor_batches = val_eval_batches[: min(8, eval_iters)]
monitor_history = []


def record_loop_metrics(model, iter):
    metrics = estimate_loop_metrics(model, monitor_batches)
    monitor_history.extend({"step": iter, **row} for row in metrics)
    print(
        f"  same-token val: 1 loop loss {metrics[0]['val_loss']:.4f}, "
        f"accuracy {metrics[0]['accuracy']:.2%}; "
        f"{n_loop} loops loss {metrics[-1]['val_loss']:.4f}, "
        f"accuracy {metrics[-1]['accuracy']:.2%}"
    )


# ===== Codex 添加结束 =====

looped_history, loop_best_step, loop_best_val = train_model(
    looped_model, "looped_gpt", record_loop_metrics
)

# ===== Codex 添加：保存训练过程中的逐轮监测结果 =====
write_metrics_csv("loop_train_metrics.csv", monitor_history)
print("Saved loop_train_metrics.csv")
# ===== Codex 添加结束 =====

# 保存完整训练曲线，便于检查 train/val 分叉和早停位置
with open("train_history.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f, fieldnames=["model", "step", "train_loss", "val_loss", "lr"]
    )
    writer.writeheader()
    writer.writerows(gpt_history + looped_history)
print("Saved train_history.csv")

torch.save(
    {
        "model_state_dict": model.state_dict(),
        "config": {
            "n_embd": n_embd,
            "n_head": n_head,
            "n_layer": n_layer,
            "block_size": block_size,
            "dropout": dropout,
        },
        "chars": chars,
        "best_step": gpt_best_step,
        "best_val_loss": gpt_best_val,
    },
    "gpt_checkpoint.pth",
)

torch.save(
    {
        "model_state_dict": looped_model.state_dict(),
        "config": {
            "n_embd": n_embd,
            "n_head": n_head,
            "n_loop": n_loop,
            "block_size": block_size,
            "dropout": dropout,
        },
        "chars": chars,
        "best_step": loop_best_step,
        "best_val_loss": loop_best_val,
    },
    "looped_gpt_checkpoint.pth",
)

# ==================== 训练结束后的验证指标 ====================
# n_loop 层是训练时的完整循环
# 普通 GPT 也在同一批数据上算一次，作为对照
print(
    "Evaluating looped_gpt with 1..n_loop loops and the normal gpt on validation set..."
)

eval_batches = val_eval_batches
# ===== Codex 添加：在同一批 token 上统计所有循环深度，并保存详细指标 =====
final_metrics = estimate_loop_metrics(looped_model, eval_batches)
write_metrics_csv("loop_token_metrics.csv", final_metrics)
loop_val_losses = {row["n_loops"]: row["val_loss"] for row in final_metrics}
# ===== Codex 添加结束 =====
gpt_val_loss = estimate_loss_on_batches(model, eval_batches)

# ===== Codex 添加：输出逐轮正确率与 token 状态变化 =====
for row in final_metrics:
    transition = (
        f", improved {row['fraction_improved']:.1%}, flat {row['fraction_flat']:.1%}, "
        f"wrong->right {row['became_correct']:.1%}, right->wrong {row['became_wrong']:.1%}"
        if row["n_loops"] > 1
        else ""
    )
    print(
        f"looped_gpt:{row['n_loops']} loops: val loss {row['val_loss']:.4f}, "
        f"accuracy {row['accuracy']:.2%}{transition}"
    )
print(f"gpt:full model: val loss {gpt_val_loss:.4f}")
print("Saved loop_token_metrics.csv")
# ===== Codex 添加结束 =====

# 保存原始数值，方便以后复查或重新画图（最后一行为普通 GPT 的对照 loss）
with open("loop_val_losses.csv", "w") as f:
    f.write("n_loops,val_loss\n")
    for n_loops, val_loss in loop_val_losses.items():
        f.write(f"{n_loops},{val_loss:.6f}\n")
    f.write(f"gpt,{gpt_val_loss:.6f}\n")
