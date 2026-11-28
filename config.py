import torch

# ==================== 训练超参数 ====================

# 1M 字符的数据集：batch 80 × 5000 步 ≈ 102 个 epoch，配合 28.5M 参数的 GPT 会严重过拟合。
# 回到 batch 32（≈41 epoch），并配合早停，实际训练步数会更少。
batch_size = 32

block_size = 256

max_iters = 5000

eval_interval = 500

learning_rate = 3e-4

min_lr = 3e-5

warmup_iters = 100

weight_decay = 0.1

grad_clip = 1.0

# 连续多少个 eval_interval 没有改善就早停；保存验证 loss 最好的 checkpoint
early_stop_patience = 2

device = "cuda" if torch.cuda.is_available() else "cpu"
print("运行平台：", device)

eval_iters = 200

# 384 时 GPT 有 28.5M 参数（约 28 参数/字符），最终 train loss 0.4 / val loss 2.23。
# 回到 256：GPT ≈12.7M，looped ≈0.89M；两边仍然都做 16 次 block 计算。
n_embd = 256

n_head = 16

# n_layer 与 n_loop 保持一致：普通 GPT 的 16 个 Block 对应共享 Block 的 16 次循环。
n_layer = 16

n_loop = 16

dropout = 0.2

torch.manual_seed(1337)
