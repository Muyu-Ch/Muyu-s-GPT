import torch

# ==================== 训练超参数 ====================

batch_size = 32

block_size = 256

max_iters = 5000

eval_interval = 500

learning_rate = 3e-4

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("运行平台：",device)

eval_iters = 200

n_embd = 256

n_head = 16

n_layer = 16

n_loop=16

dropout = 0.2

torch.manual_seed(1337)
