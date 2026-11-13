import matplotlib
matplotlib.use('Agg')  # 不弹窗，直接把图保存成文件
import matplotlib.pyplot as plt

from config import n_layer

print("正在读取 loop_val_losses.csv ...")

loop_val_losses = {}
gpt_val_loss = None

with open('loop_val_losses.csv', 'r') as f:
    f.readline()  # 跳过表头 n_loops,val_loss
    for line in f:
        name, value = line.strip().split(',')
        if name == 'gpt':
            gpt_val_loss = float(value)
        else:
            loop_val_losses[int(name)] = float(value)

if gpt_val_loss is None:
    raise ValueError("loop_val_losses.csv 中缺少普通 gpt 的对照 loss（最后一行应以 gpt, 开头）")

# ==================== 生图部分：验证 loss 随循环层数变化的折线图 ====================
counts = list(loop_val_losses.keys())
losses = [loop_val_losses[n] for n in counts]
best_n = min(counts, key=lambda n: loop_val_losses[n])
n_loop = counts[-1]

print(f"读取到 {len(counts)} 个循环层数的结果，开始画图 ...")

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
