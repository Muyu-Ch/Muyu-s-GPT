"""把逐循环深度的监测结果画成两张可读的分析图。

输入：
  loop_token_metrics.csv   最终模型在验证集上的逐深度指标
  loop_train_metrics.csv   训练过程中每个 checkpoint 在固定样本上的逐深度指标
  loop_val_losses.csv      普通 GPT 的最终 val loss（对照）

输出：
  loop_depth_analysis.png    最终模型的深度分析（loss / 正确率 / 纠错动态 / 难度分组）
  loop_training_analysis.png 训练过程（loss 热力图 + 关键深度曲线）

图中的标题、最优点、标注数值都从 CSV 现算，重新训练后直接重跑即可。
"""

import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D
from matplotlib.patheffects import withStroke
from matplotlib.ticker import MaxNLocator

# ==================== 设计参数（dataviz skill 的验证调色板） ====================
SURFACE = "#ffffff"
INK = "#191919"
INK_2 = "#666666"
AXIS = "#c2c2c2"
GRID = "#e7e7e7"
BLUE = "#257de0"
RED = "#ec4647"
GRAY = "#898781"

# 顺序蓝：q1→q4、1→16 轮等有序变量
ORDINAL_4 = ["#84b9f3", "#519ced", "#2370cd", "#0e3d7e"]
ORDINAL_5 = ["#84b9f3", "#519ced", "#2370cd", "#1757ac", "#0e3d7e"]
SEQUENTIAL = ["#cce3fd", "#9dc8f7", "#6aabf1", "#338ced", "#2370cd", "#1757ac", "#0e3d7e"]

plt.rcParams.update({
    "font.family": "Hiragino Sans GB",
    "font.size": 9.5,
    "axes.unicode_minus": False,
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "axes.edgecolor": AXIS,
    "axes.linewidth": 0.8,
    "axes.labelcolor": INK_2,
    "text.color": INK,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "legend.frameon": False,
})


def style_axis(ax, ygrid=True):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    if ygrid:
        ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=GRAY, labelsize=8.5, length=3, width=0.8)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_color(INK_2)


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def fnum(row, key):
    value = row.get(key)
    return float(value) if value not in (None, "", "nan") else float("nan")


# ==================== 数据 ====================
final_rows = read_csv("loop_token_metrics.csv")
train_rows = read_csv("loop_train_metrics.csv")

depths = np.array([int(r["n_loops"]) for r in final_rows])
loss = np.array([fnum(r, "val_loss") for r in final_rows])
accuracy = np.array([fnum(r, "accuracy") for r in final_rows]) * 100

best_loss_depth = int(depths[loss.argmin()])
trained_depth = int(depths[-1])
best_acc_depth = int(depths[accuracy.argmax()])

# 条件翻转率：分母是上一轮对应的 token 数，比原始占比更好读
prev_acc = accuracy / 100
wrong_to_right = np.array([
    fnum(r, "became_correct") / (1 - prev_acc[i - 1]) if i else np.nan
    for i, r in enumerate(final_rows)
]) * 100
right_to_wrong = np.array([
    fnum(r, "became_wrong") / prev_acc[i - 1] if i else np.nan
    for i, r in enumerate(final_rows)
]) * 100

# 四组 token 相对各自最优 loss 的增量
q_keys = [f"initial_loss_q{i}" for i in range(1, 5)]
q_loss = np.array([[fnum(r, key) for r in final_rows] for key in q_keys])
q_regret = q_loss - q_loss.min(axis=1, keepdims=True)
q_best = q_loss.argmin(axis=1) + 1
q_best_label = []
for i in range(4):
    tied = depths[q_loss[i] <= q_loss[i].min() + 1e-3]
    q_best_label.append(
        str(int(tied.min())) if tied.min() == tied.max() else f"{int(tied.min())}–{int(tied.max())}"
    )

# 普通 GPT 基线
with open("loop_val_losses.csv", newline="", encoding="utf-8") as f:
    gpt_loss = next(float(r["val_loss"]) for r in csv.DictReader(f) if r["n_loops"] == "gpt")

# 训练过程中的逐深度 loss（去掉 step 0 的随机初始化，避免压缩色标）
steps = sorted({int(r["step"]) for r in train_rows if int(r["step"]) > 0})
step_labels = [str(s) for s in steps]
grid = np.full((len(depths), len(steps)), np.nan)
for row in train_rows:
    step = int(row["step"])
    if step in steps:
        grid[int(row["n_loops"]) - 1, steps.index(step)] = fnum(row, "val_loss")
best_depth_by_step = np.nanargmin(grid, axis=0) + 1

# 共同拐点：用较深的一半深度的最优 step 的中位数
deep_start = max(1, len(depths) - 8)
deep_best_steps = [
    steps[int(np.nanargmin(grid[depth - 1]))]
    for depth in range(deep_start, len(depths) + 1)
]
turn_step = int(np.median(deep_best_steps))
best16_idx = int(np.nanargmin(grid[-1]))
best16_step = steps[best16_idx]
best16_loss = grid[-1, best16_idx]

# ==================== 图 1：最终模型的循环深度分析 ====================
fig, axes = plt.subplots(2, 2, figsize=(12.6, 9.6))
fig.subplots_adjust(left=0.065, right=0.975, top=0.865, bottom=0.075, hspace=0.42, wspace=0.24)

fig.text(0.065, 0.965, "循环深度分析：收益、饱和与过循环", fontsize=17, fontweight="semibold",
         color=INK, ha="left", va="top")
fig.text(0.065, 0.925,
         f"最终模型，完整验证集。1 个共享 Block 重复 1–{trained_depth} 次；"
         f"1 → {best_loss_depth} 轮 loss {loss[0]:.2f} → {loss.min():.2f}，"
         f"1 → {best_acc_depth} 轮正确率 {accuracy[0]:.1f}% → {accuracy.max():.1f}%。",
         fontsize=9.8, color=INK_2, ha="left", va="top")

# --- A. loss vs depth ---
ax = axes[0, 0]
y_lo = min(loss.min(), gpt_loss) - 0.10
y_hi = max(loss.max(), gpt_loss) + 0.12
ax.set_ylim(y_lo, y_hi)
if trained_depth > best_loss_depth:
    ax.axvspan(best_loss_depth + 0.5, trained_depth + 0.5, color="#f4f4f2", zorder=0)
    ax.text((best_loss_depth + trained_depth) / 2 + 0.5, y_hi - 0.02, "过循环区",
            fontsize=8.8, color=GRAY, ha="center", va="top")
ax.axhline(gpt_loss, color=GRAY, linewidth=1.4, zorder=2)
gpt_label_x = 9.3
gpt_label_y = gpt_loss + 0.035
gpt_label_va = "bottom"
if gpt_loss < np.interp(gpt_label_x, depths, loss):
    gpt_label_y = gpt_loss - 0.035
    gpt_label_va = "top"
ax.text(gpt_label_x, gpt_label_y, f"普通 GPT（16 层）{gpt_loss:.3f}",
        fontsize=8.8, color=INK_2, ha="left", va=gpt_label_va)
ax.plot(depths, loss, color=BLUE, linewidth=2.0, solid_capstyle="round", zorder=3)
ax.plot(depths, loss, "o", color=BLUE, markersize=4.0, markeredgecolor=SURFACE,
        markeredgewidth=1.1, zorder=4)
ax.plot(depths[loss.argmin()], loss.min(), "o", color=BLUE, markersize=8.5,
        markeredgecolor=SURFACE, markeredgewidth=2.0, zorder=5)
ax.plot(trained_depth, loss[-1], "o", markerfacecolor=SURFACE, markeredgecolor=BLUE,
        markersize=8.5, markeredgewidth=2.0, zorder=5)
ax.annotate(f"最优 {best_loss_depth} 轮 · {loss.min():.3f}",
            xy=(best_loss_depth, loss.min()), xytext=(2.0, y_lo + 0.04),
            fontsize=9.2, color=INK, ha="left", va="bottom",
            arrowprops=dict(arrowstyle="-", color=AXIS, linewidth=0.8))
ax.annotate(f"训练深度 {trained_depth} 轮 · {loss[-1]:.3f}",
            xy=(trained_depth, loss[-1]), xytext=(trained_depth - 0.4, loss[-1] + 0.16),
            fontsize=9.2, color=INK, ha="right", va="bottom",
            arrowprops=dict(arrowstyle="-", color=AXIS, linewidth=0.8))
ax.set_xlim(0.6, trained_depth + 0.4)
ax.set_xticks(depths)
ax.set_xlabel("循环深度（同一个 Block 复用次数）", fontsize=9.5)
ax.set_ylabel("验证交叉熵 loss", fontsize=9.5)
if best_loss_depth == trained_depth:
    loss_tail = "训练深度即最优"
elif loss[-1] > loss.min():
    loss_tail = f"{trained_depth} 轮反而更差"
else:
    loss_tail = "之后基本持平"
ax.set_title(f"A  loss：{best_loss_depth} 轮见底，{loss_tail}", loc="left", fontsize=11.5,
             fontweight="semibold", pad=8)
style_axis(ax)

# --- B. accuracy vs depth ---
ax = axes[0, 1]
acc_top = max(62.0, accuracy.max() + 6)
ax.set_ylim(20, acc_top)
ax.plot(depths, accuracy, color=BLUE, linewidth=2.0, solid_capstyle="round", zorder=3)
ax.plot(depths, accuracy, "o", color=BLUE, markersize=4.0, markeredgecolor=SURFACE,
        markeredgewidth=1.1, zorder=4)
ax.plot(best_acc_depth, accuracy.max(), "o", color=BLUE, markersize=8.5,
        markeredgecolor=SURFACE, markeredgewidth=2.0, zorder=5)
ax.plot(trained_depth, accuracy[-1], "o", markerfacecolor=SURFACE, markeredgecolor=BLUE,
        markersize=8.5, markeredgewidth=2.0, zorder=5)
ax.annotate(f"{accuracy[0]:.1f}%", xy=(1, accuracy[0]), xytext=(1.35, accuracy[0] + 1.8),
            fontsize=9.2, color=INK, ha="left", va="bottom")
ax.annotate(f"最优 {best_acc_depth} 轮 {accuracy.max():.1f}%\n{trained_depth} 轮基本持平",
            xy=(best_acc_depth, accuracy.max()),
            xytext=(best_acc_depth - 2.5, acc_top - 2.0),
            fontsize=9.2, color=INK, ha="right", va="top",
            arrowprops=dict(arrowstyle="-", color=AXIS, linewidth=0.8))
gain_target = accuracy[0] + 0.9 * (accuracy.max() - accuracy[0])
n90 = int(depths[np.argmax(accuracy >= gain_target)])
ax.text(1.8, accuracy.max() - 1.0, f"前 {n90} 轮完成约 90% 的正确率增益",
        fontsize=8.8, color=INK_2, ha="left", va="top")
ax.set_xlim(0.6, trained_depth + 0.4)
ax.set_xticks(depths)
ax.set_xlabel("循环深度（同一个 Block 复用次数）", fontsize=9.5)
ax.set_ylabel("top-1 正确率 (%)", fontsize=9.5)
if best_acc_depth == trained_depth:
    acc_tail = "训练深度即最优"
else:
    acc_tail = "之后不再提升"
ax.set_title(f"B  正确率：{best_acc_depth} 轮见顶，{acc_tail}", loc="left", fontsize=11.5,
             fontweight="semibold", pad=8)
style_axis(ax)

# --- C. 条件翻转率 ---
ax = axes[1, 0]
ax.plot(depths[1:], wrong_to_right[1:], color=BLUE, linewidth=2.0, solid_capstyle="round",
        label="错误 → 正确", zorder=3)
ax.plot(depths[1:], wrong_to_right[1:], "o", color=BLUE, markersize=4.0,
        markeredgecolor=SURFACE, markeredgewidth=1.1, zorder=4)
ax.plot(depths[1:], right_to_wrong[1:], color=RED, linewidth=2.0, solid_capstyle="round",
        label="正确 → 错误", zorder=3)
ax.plot(depths[1:], right_to_wrong[1:], "o", color=RED, markersize=4.0,
        markeredgecolor=SURFACE, markeredgewidth=1.1, zorder=4)

ratios = wrong_to_right[1:] / right_to_wrong[1:]
peak_i = int(np.nanargmax(ratios))
peak_depth = int(depths[1:][peak_i])
peak_ratio = ratios[peak_i]
ax.text(0.02, 0.97, f"峰值：第 {peak_depth} 轮，纠错是破坏的 {peak_ratio:.1f} 倍",
        transform=ax.transAxes, fontsize=8.8, color=INK_2, ha="left", va="top")
last_ratio = wrong_to_right[-1] / right_to_wrong[-1]
ax.annotate(f"最后一轮：纠错/破坏 = {last_ratio:.1f}",
            xy=(depths[-1], right_to_wrong[-1]), xytext=(depths[-1] - 0.5, 0.45),
            fontsize=8.8, color=INK_2, ha="right", va="bottom",
            arrowprops=dict(arrowstyle="-", color=AXIS, linewidth=0.8))
ax.set_xlim(1.6, trained_depth + 0.5)
ax.set_ylim(0, max(10.6, np.nanmax(wrong_to_right) + 0.8))
ax.set_xticks(range(2, trained_depth + 1, 2))
ax.set_xlabel("循环深度", fontsize=9.5)
ax.set_ylabel("占上一轮对应 token 的比例 (%)", fontsize=9.5)
ax.set_title("C  每一轮是在纠错，还是在制造新错误？", loc="left", fontsize=11.5,
             fontweight="semibold", pad=8)
ax.legend(loc="upper right", fontsize=8.8, handlelength=1.8, borderaxespad=0.2)
style_axis(ax)

# --- D. 四组难度的相对 loss 增量 ---
ax = axes[1, 1]
q_names = ["q1 最简单", "q2", "q3", "q4 最难"]
handles = []
for i, (name, color) in enumerate(zip(q_names, ORDINAL_4)):
    ax.plot(depths, q_regret[i], color=color, linewidth=2.0, solid_capstyle="round", zorder=3)
    ax.plot(depths, q_regret[i], "o", color=color, markersize=3.6, markeredgecolor=SURFACE,
            markeredgewidth=0.9, zorder=4)
    ax.plot(q_best[i], q_regret[i][q_best[i] - 1], "o", color=color, markersize=7.5,
            markeredgecolor=SURFACE, markeredgewidth=1.8, zorder=5)
    handles.append(Line2D([0], [0], color=color, linewidth=2.0,
                          label=f"{name} · 最优 {q_best_label[i]} 轮"))
ax.set_xlim(0.6, trained_depth + 0.4)
ax.set_ylim(-0.10, max(2.10, float(np.nanmax(q_regret)) + 0.10))
ax.set_xticks(depths)
ax.set_xlabel("循环深度", fontsize=9.5)
ax.set_ylabel("相对该组自身最优的 loss 增量", fontsize=9.5)
ax.set_title("D  越简单的 token 越早饱和", loc="left", fontsize=11.5,
             fontweight="semibold", pad=8)
ax.legend(handles=handles, loc="upper right", fontsize=8.5, handlelength=1.8,
          borderaxespad=0.2, labelspacing=0.35)
style_axis(ax)

fig.text(0.065, 0.022, "数据：loop_token_metrics.csv；正确率 = 下一字符 top-1；"
                       "条件翻转率以「上一轮对/错」为分母；q1–q4 按第 1 轮 loss 从低到高固定分组。",
         fontsize=8.2, color=GRAY, ha="left", va="bottom")
fig.savefig("loop_depth_analysis.png", dpi=200)
plt.close(fig)
print("Saved loop_depth_analysis.png")

# ==================== 图 2：训练过程 ====================
fig, axes = plt.subplots(1, 2, figsize=(14.2, 5.6),
                         gridspec_kw={"width_ratios": [1.15, 1.0]})
fig.subplots_adjust(left=0.055, right=0.975, top=0.745, bottom=0.135, wspace=0.22)

fig.text(0.055, 0.955, "训练过程：循环能力怎么长出来的", fontsize=17, fontweight="semibold",
         color=INK, ha="left", va="top")
fig.text(0.055, 0.900,
         "固定 8 批验证样本；每个 checkpoint 在同一批 token 上评估 1–16 轮。\n"
         f"热力图深色 = loss 更高，细线 = 每步最优深度（{best_depth_by_step[0]} → {best_depth_by_step[-1]}）；"
         "深层曲线在同一个拐点见底，1 轮从更早开始变差。",
         fontsize=9.4, color=INK_2, ha="left", va="top", linespacing=1.7)

# --- A. 热力图 ---
ax = axes[0]
cmap = LinearSegmentedColormap.from_list("seq_blue", SEQUENTIAL)
vmin = float(np.nanmin(grid))
vmax = float(np.nanmax(grid))
if vmax - vmin < 1e-6:
    vmax = vmin + 1e-3
norm = Normalize(vmin=vmin, vmax=vmax)
im = ax.imshow(grid, aspect="auto", cmap=cmap, norm=norm, origin="lower",
               interpolation="nearest")
ridge = best_depth_by_step - 1
ax.plot(range(len(steps)), ridge, color=INK, linewidth=1.4, marker="o", markersize=3.4,
        markeredgecolor=SURFACE, markeredgewidth=0.7, zorder=3,
        path_effects=[withStroke(linewidth=3.0, foreground=SURFACE)])
ax.set_xticks(range(len(steps)))
ax.set_xticklabels(step_labels, fontsize=8.0)
ax.set_yticks(range(len(depths)))
ax.set_yticklabels([str(d) for d in depths], fontsize=7.8)
ax.set_xlabel("训练步数", fontsize=9.5)
ax.set_ylabel("循环深度", fontsize=9.5)
ax.set_title("A  每个 checkpoint 的 val loss", loc="left",
             fontsize=11.5, fontweight="semibold", pad=8)
ax.tick_params(colors=GRAY, length=2.5, width=0.8)
for label in ax.get_xticklabels() + ax.get_yticklabels():
    label.set_color(INK_2)
for spine in ax.spines.values():
    spine.set_visible(False)
cbar = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.018)
cbar.outline.set_visible(False)
cbar.ax.tick_params(colors=GRAY, labelsize=8.0, length=2.5)
for label in cbar.ax.get_yticklabels():
    label.set_color(INK_2)
cbar.set_label("val loss", color=INK_2, fontsize=9.0)
cbar.locator = MaxNLocator(nbins=4)
cbar.update_ticks()

# --- B. 关键深度曲线 ---
ax = axes[1]
line_depths = [1, 4, 8, 12, 16]
line_depths = [d for d in line_depths if d <= trained_depth]
for depth, color in zip(line_depths, ORDINAL_5):
    series = grid[depth - 1]
    ax.plot(steps, series, color=color, linewidth=1.9, solid_capstyle="round",
            marker="o", markersize=3.2, markeredgecolor=SURFACE, markeredgewidth=0.7,
            label=f"{depth} 轮", zorder=3)

y_lo = max(0.0, float(np.nanmin(grid)) - 0.08)
y_hi = float(np.nanmax(grid)) + 0.22
ax.set_ylim(y_lo, y_hi)
ax.axvspan(turn_step, steps[-1] + 300, color="#f4f4f2", zorder=0)
ax.axvline(turn_step, color=AXIS, linewidth=0.8, zorder=1)
ax.plot(best16_step, best16_loss, "o", color=ORDINAL_5[-1], markersize=8.0,
        markeredgecolor=SURFACE, markeredgewidth=2.0, zorder=5)
ax.annotate(f"{best16_step} 步见底 {best16_loss:.3f}",
            xy=(best16_step, best16_loss),
            xytext=(max(steps[0] + 50, best16_step - 550), y_lo + 0.035),
            fontsize=8.8, color=INK, ha="right", va="bottom",
            arrowprops=dict(arrowstyle="-", color=AXIS, linewidth=0.8))
ax.text((turn_step + steps[-1]) / 2, y_lo + 0.02, "拐点之后：loss 回升",
        fontsize=8.8, color=GRAY, ha="center", va="bottom")
if grid[0, -1] > grid[0, 0]:
    ax.annotate("1 轮随训练变差", xy=(steps[-1], grid[0, -1]),
                xytext=(steps[-1] - 700, grid[0, -1] - 0.12),
                fontsize=8.8, color=INK, ha="right", va="top",
                arrowprops=dict(arrowstyle="-", color=AXIS, linewidth=0.8))
ax.annotate(f"{trained_depth} 轮", xy=(steps[-1], grid[-1, -1]),
            xytext=(steps[-1] + 120, grid[-1, -1]), fontsize=8.8, color=INK,
            ha="left", va="bottom")
ax.set_xlim(steps[0] - 200, steps[-1] + 320)
ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
ax.set_xlabel("训练步数", fontsize=9.5)
ax.set_ylabel("验证 loss", fontsize=9.5)
ax.set_title(f"B  五个深度：深层在 {turn_step} 步见底", loc="left", fontsize=11.5,
             fontweight="semibold", pad=8)
ax.legend(loc="upper right", fontsize=8.5, ncol=5, handlelength=1.3,
          columnspacing=0.9, borderaxespad=0.2)
style_axis(ax)

fig.text(0.055, 0.035, "数据：loop_train_metrics.csv；热力图只画训练开始后的 checkpoint，"
                       "避免初始高 loss 压缩色标。",
         fontsize=8.2, color=GRAY, ha="left", va="bottom")
fig.savefig("loop_training_analysis.png", dpi=200)
plt.close(fig)
print("Saved loop_training_analysis.png")
