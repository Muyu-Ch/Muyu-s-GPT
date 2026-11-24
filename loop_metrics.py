"""同一批验证 token 在不同循环深度下的 loss 与 top-1 正确率。"""

# ===== Codex 添加：本文件为逐 token 循环深度监测功能，用于验证用户的实验猜想 =====

import csv

import torch
from torch.nn import functional as F


@torch.no_grad()
def estimate_loop_metrics(model, batches, loss_change_threshold=0.01):
    was_training = model.training
    model.eval()
    losses_by_depth = [[] for _ in range(model.n_loop)]
    correct_by_depth = [[] for _ in range(model.n_loop)]
    try:
        for inputs, targets in batches:
            targets = targets.reshape(-1)
            for depth, logits in enumerate(model.logits_at_each_loop(inputs)):
                logits = logits.reshape(-1, logits.shape[-1])
                losses_by_depth[depth].append(
                    F.cross_entropy(logits, targets, reduction='none').cpu()
                )
                correct_by_depth[depth].append(logits.argmax(-1).eq(targets).cpu())
    finally:
        model.train(was_training)

    losses = torch.stack([torch.cat(parts) for parts in losses_by_depth])
    correct = torch.stack([torch.cat(parts) for parts in correct_by_depth])
    # 第 1 轮的 loss 将 token 分成四组；之后各轮保持组成员不变。
    quartiles = torch.quantile(losses[0], torch.tensor([0.25, 0.5, 0.75]))
    groups = torch.bucketize(losses[0], quartiles)
    rows = []
    for depth in range(model.n_loop):
        token_loss, token_correct = losses[depth], correct[depth]
        row = {
            'n_loops': depth + 1,
            'val_loss': token_loss.mean().item(),
            'accuracy': token_correct.float().mean().item(),
            'loss_p10': torch.quantile(token_loss, 0.1).item(),
            'loss_p50': torch.quantile(token_loss, 0.5).item(),
            'loss_p90': torch.quantile(token_loss, 0.9).item(),
        }
        for group in range(4):
            selected = token_loss[groups == group]
            row[f'initial_loss_q{group + 1}'] = selected.mean().item() if selected.numel() else float('nan')
        if depth:
            gain = losses[depth - 1] - token_loss
            previous_correct = correct[depth - 1]
            row.update({
                'fraction_improved': (gain > loss_change_threshold).float().mean().item(),
                'fraction_flat': (gain.abs() <= loss_change_threshold).float().mean().item(),
                'fraction_worsened': (gain < -loss_change_threshold).float().mean().item(),
                'became_correct': ((~previous_correct) & token_correct).float().mean().item(),
                'became_wrong': (previous_correct & (~token_correct)).float().mean().item(),
                'gain_prev_correct': gain[previous_correct].mean().item() if previous_correct.any() else float('nan'),
                'gain_prev_wrong': gain[~previous_correct].mean().item() if (~previous_correct).any() else float('nan'),
            })
        rows.append(row)
    return rows


def write_metrics_csv(path, rows):
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with open(path, 'w', newline='', encoding='utf-8') as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
# ===== Codex 添加结束 =====
