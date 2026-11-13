import torch

from config import device
from model import GPTLanguageModel, LoopedGPTLanguageModel

print("正在加载 gpt_checkpoint.pth ...")

gpt_checkpoint = torch.load('gpt_checkpoint.pth', map_location=device)
gpt_config = gpt_checkpoint['config']
chars = gpt_checkpoint['chars']

itos = { i:ch for i,ch in enumerate(chars) }

decode = lambda l: ''.join([itos[i] for i in l])

print("正在根据 checkpoint 创建 gpt 模型 ...")

model = GPTLanguageModel(
    vocab_size=len(chars),
    n_embd=gpt_config['n_embd'],
    n_head=gpt_config['n_head'],
    n_layer=gpt_config['n_layer'],
    block_size=gpt_config['block_size'],
    dropout=gpt_config['dropout'],
)
model.load_state_dict(gpt_checkpoint['model_state_dict'])
model = model.to(device)

print("正在加载 looped_gpt_checkpoint.pth ...")

looped_checkpoint = torch.load('looped_gpt_checkpoint.pth', map_location=device)
looped_config = looped_checkpoint['config']
looped_chars = looped_checkpoint['chars']

looped_itos = { i:ch for i,ch in enumerate(looped_chars) }

looped_decode = lambda l: ''.join([looped_itos[i] for i in l])

print("正在根据 checkpoint 创建 looped_gpt 模型 ...")

looped_model = LoopedGPTLanguageModel(
    vocab_size=len(looped_chars),
    n_embd=looped_config['n_embd'],
    n_head=looped_config['n_head'],
    n_loop=looped_config['n_loop'],
    block_size=looped_config['block_size'],
    dropout=looped_config['dropout'],
)
looped_model.load_state_dict(looped_checkpoint['model_state_dict'])
looped_model = looped_model.to(device)

print("Generating text from gpt model...")

context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(model.generate(context, max_new_tokens=500)[0].tolist()))
open('more.txt', 'w').write(decode(model.generate(context, max_new_tokens=10000)[0].tolist()))

print("Generating text from looped_gpt model (full n_loop loops)...")

lmcontext = torch.zeros((1, 1), dtype=torch.long, device=device)
print(looped_decode(looped_model.generate(lmcontext, max_new_tokens=500)[0].tolist()))
open('looped_more.txt', 'w').write(looped_decode(looped_model.generate(lmcontext, max_new_tokens=10000)[0].tolist()))

# 用 4、8、12 层循环生成文本，输出格式与原来的版本一致
for n_loops in [4, 8, 12]:
    print(f"Generating text from less_looped_gpt model ({n_loops} loops)...")

    less_lmcontext = torch.zeros((1, 1), dtype=torch.long, device=device)
    print(looped_decode(looped_model.generate_n_loops(less_lmcontext, max_new_tokens=500, n_loops=n_loops)[0].tolist()))
    open(f'less_looped_{n_loops}_more.txt', 'w').write(
        looped_decode(looped_model.generate_n_loops(less_lmcontext, max_new_tokens=10000, n_loops=n_loops)[0].tolist())
    )

print("generate.py 全部完成，文本已保存到 more.txt、looped_more.txt、less_looped_*_more.txt")
