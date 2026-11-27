"""把逐循环深度的监测结果画成两张可读的分析图。

输入：
  loop_token_metrics.csv   最终模型在 200 批验证样本上的逐深度指标
  loop_train_metrics.csv   训练过程中每 500 步在固定 8 批样本上的逐深度指标
  loop_val_losses.csv      普通 GPT 的最终 val loss（对照）

输出：
  loop_depth_analysis.png    最终模型的深度分析（loss / 正确率 / 纠错动态 / 难度分组）
  loop_training_analysis.png 训练过程（loss 热力图 + 关键深度曲线）
"""

import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D
from matplotlib.patheffects import withStroke

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

# 普通 GPT 基线
with open("loop_val_losses.csv", newline="", encoding="utf-8") as f:
    gpt_loss = next(float(r["val_loss"]) for r in csv.DictReader(f) if r["n_loops"] == "gpt")

steps = sorted({int(r["step"]) for r in train_rows if int(r["step"]) >= 500})
step_labels = [str(s) for s in steps]
grid = np.full((len(depths), len(steps)), np.nan)
for row in train_rows:
    step = int(row["step"])
    if step in steps:
        grid[int(row["n_loops"]) - 1, steps.index(step)] = fnum(row, "val_loss")
best_depth_by_step = np.nanargmin(grid, axis=0) + 1

# ==================== 图 1：最终模型的循环深度分析 ====================
fig, axes = plt.subplots(2, 2, figsize=(12.6, 9.6))
fig.subplots_adjust(left=0.065, right=0.975, top=0.865, bottom=0.075, hspace=0.42, wspace=0.24)

fig.text(0.065, 0.965, "循环深度分析：收益、饱和与过循环", fontsize=17, fontweight="semibold",
         color=INK, ha="left", va="top")
fig.text(0.065, 0.925,
         "最终模型，200 批验证样本。1 个共享 Block 重复 1–16 次；1 → 13 轮 loss 2.73 → 1.52，"
         "1 → 15 轮正确率 23.5% → 56.7%。",
         fontsize=9.8, color=INK_2, ha="left", va="top")

# --- A. loss vs depth ---
ax = axes[0, 0]
best_loss_depth = int(depths[loss.argmin()])
trained_depth = int(depths[-1])
ax.axvspan(best_loss_depth + 0.5, trained_depth + 0.5, color="#f4f4f2", zorder=0)
ax.axhline(gpt_loss, color=GRAY, linewidth=1.4, zorder=2)
ax.text(9.3, gpt_loss + 0.035, f"普通 GPT（16 层）{gpt_loss:.3f}", fontsize=8.8, color=INK_2,
        ha="left", va="bottom")
ax.plot(depths, loss, color=BLUE, linewidth=2.0, solid_capstyle="round", zorder=3)
ax.plot(depths, loss, "o", color=BLUE, markersize=4.0, markeredgecolor=SURFACE,
        markeredgewidth=1.1, zorder=4)
ax.plot(depths[loss.argmin()], loss.min(), "o", color=BLUE, markersize=8.5,
        markeredgecolor=SURFACE, markeredgewidth=2.0, zorder=5)
ax.plot(trained_depth, loss[-1], "o", markerfacecolor=SURFACE, markeredgecolor=BLUE,
        markersize=8.5, markeredgewidth=2.0, zorder=5)
ax.annotate(f"最优 {best_loss_depth} 轮 · {loss.min():.3f}",
            xy=(best_loss_depth, loss.min()), xytext=(2.0, 1.395),
            fontsize=9.2, color=INK, ha="left", va="bottom",
            arrowprops=dict(arrowstyle="-", color=AXIS, linewidth=0.8))
ax.annotate(f"训练深度 {trained_depth} 轮 · {loss[-1]:.3f}",
            xy=(trained_depth, loss[-1]), xytext=(trained_depth - 0.4, loss[-1] + 0.16),
            fontsize=9.2, color=INK, ha="right", va="bottom",
            arrowprops=dict(arrowstyle="-", color=AXIS, linewidth=0.8))
ax.text(best_loss_depth + 1.55, 2.60, "过循环区", fontsize=8.8, color=GRAY, ha="center")
ax.set_xlim(0.6, 16.4)
ax.set_ylim(1.35, 2.82)
ax.set_xticks(depths)
ax.set_xlabel("循环深度（同一个 Block 复用次数）", fontsize=9.5)
ax.set_ylabel("验证交叉熵 loss", fontsize=9.5)
ax.set_title("A  loss：13 轮见底，16 轮反而更差", loc="left", fontsize=11.5,
             fontweight="semibold", pad=8)
style_axis(ax)

# --- B. accuracy vs depth ---
ax = axes[0, 1]
best_acc_depth = int(depths[accuracy.argmax()])
ax.plot(depths, accuracy, color=BLUE, linewidth=2.0, solid_capstyle="round", zorder=3)
ax.plot(depths, accuracy, "o", color=BLUE, markersize=4.0, markeredgecolor=SURFACE,
        markeredgewidth=1.1, zorder=4)
ax.plot(best_acc_depth, accuracy.max(), "o", color=BLUE, markersize=8.5,
        markeredgecolor=SURFACE, markeredgewidth=2.0, zorder=5)
ax.plot(trained_depth, accuracy[-1], "o", markerfacecolor=SURFACE, markeredgecolor=BLUE,
        markersize=8.5, markeredgewidth=2.0, zorder=5)
ax.annotate(f"{accuracy[0]:.1f}%", xy=(1, accuracy[0]), xytext=(1.35, accuracy[0] + 1.8),
            fontsize=9.2, color=INK, ha="left", va="bottom")
ax.annotate(f"最优 {best_acc_depth} 轮 {accuracy.max():.1f}%\n16 轮基本持平",
            xy=(best_acc_depth, accuracy.max()), xytext=(12.5, 60.2),
            fontsize=9.2, color=INK, ha="right", va="top",
            arrowprops=dict(arrowstyle="-", color=AXIS, linewidth=0.8))
ax.text(1.8, 56.2, "前 9 轮完成约 89% 的正确率增益", fontsize=8.8, color=INK_2, ha="left")
ax.set_xlim(0.6, 16.4)
ax.set_ylim(20, 62)
ax.set_xticks(depths)
ax.set_xlabel("循环深度（同一个 Block 复用次数）", fontsize=9.5)
ax.set_ylabel("top-1 正确率 (%)", fontsize=9.5)
ax.set_title("B  正确率：15 轮见顶，之后不再提升", loc="left", fontsize=11.5,
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
ax.annotate("纠错约为破坏的 3 倍", xy=(6, 8.44), xytext=(7.1, 5.9), fontsize=8.8, color=INK_2,
            ha="left", va="center",
            arrowprops=dict(arrowstyle="-", color=AXIS, linewidth=0.8))
ax.annotate("15–16 轮接近持平", xy=(16, 1.28), xytext=(12.4, 0.45), fontsize=8.8, color=INK_2,
            ha="right", va="bottom",
            arrowprops=dict(arrowstyle="-", color=AXIS, linewidth=0.8))
ax.set_xlim(1.6, 16.5)
ax.set_ylim(0, 10.6)
ax.set_xticks(range(2, 17, 2))
ax.set_xlabel("循环深度", fontsize=9.5)
ax.set_ylabel("占上一轮对应 token 的比例 (%)", fontsize=9.5)
ax.set_title("C  每一轮是在纠错，还是在制造新错误？", loc="left", fontsize=11.5,
             fontweight="semibold", pad=8)
ax.legend(loc="upper right", fontsize=8.8, handlelength=1.8, borderaxespad=0.2)
style_axis(ax)

# --- D. 四组难度的相对 loss 增量 ---
ax = axes[1, 1]
q_names = ["q1 最简单", "q2", "q3", "q4 最难"]
q_best_label = ["9", "11–12", "12", "14–15"]
handles = []
for i, (name, color) in enumerate(zip(q_names, ORDINAL_4)):
    ax.plot(depths, q_regret[i], color=color, linewidth=2.0, solid_capstyle="round", zorder=3)
    ax.plot(depths, q_regret[i], "o", color=color, markersize=3.6, markeredgecolor=SURFACE,
            markeredgewidth=0.9, zorder=4)
    ax.plot(q_best[i], q_regret[i][q_best[i] - 1], "o", color=color, markersize=7.5,
            markeredgecolor=SURFACE, markeredgewidth=1.8, zorder=5)
    handles.append(Line2D([0], [0], color=color, linewidth=2.0,
                          label=f"{name} · 最优 {q_best_label[i]} 轮"))
ax.set_xlim(0.6, 16.4)
ax.set_ylim(-0.10, 2.10)
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
         "热力图深色 = loss 更高，细线 = 每步最优深度（16 → 12）；"
         "step 3000 后所有深度回升，最后 2000 步基本是过拟合。",
         fontsize=9.4, color=INK_2, ha="left", va="top", linespacing=1.7)

# --- A. 热力图 ---
ax = axes[0]
cmap = LinearSegmentedColormap.from_list("seq_blue", SEQUENTIAL)
norm = Normalize(vmin=1.45, vmax=2.75)
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
cbar.set_ticks([1.5, 2.0, 2.5])

# --- B. 关键深度曲线 ---
ax = axes[1]
line_depths = [1, 4, 8, 12, 16]
for depth, color in zip(line_depths, ORDINAL_5):
    series = grid[depth - 1]
    ax.plot(steps, series, color=color, linewidth=1.9, solid_capstyle="round",
            marker="o", markersize=3.2, markeredgecolor=SURFACE, markeredgewidth=0.7,
            label=f"{depth} 轮", zorder=3)
ax.axvspan(3000, 5350, color="#f4f4f2", zorder=0)
ax.axvline(3000, color=AXIS, linewidth=0.8, zorder=1)
ax.plot(3000, grid[15, steps.index(3000)], "o", color=ORDINAL_5[-1], markersize=8.0,
        markeredgecolor=SURFACE, markeredgewidth=2.0, zorder=5)
ax.annotate(f"3000 步见底 {grid[15, steps.index(3000)]:.3f}",
            xy=(3000, grid[15, steps.index(3000)]), xytext=(2450, 1.425),
            fontsize=8.8, color=INK, ha="right", va="bottom",
            arrowprops=dict(arrowstyle="-", color=AXIS, linewidth=0.8))
ax.text(4200, 1.415, "最后 2000 步：loss 回升", fontsize=8.8, color=GRAY, ha="center")
ax.annotate("1 轮随训练变差", xy=(4999, grid[0, -1]), xytext=(4300, 2.60),
            fontsize=8.8, color=INK, ha="right", va="bottom",
            arrowprops=dict(arrowstyle="-", color=AXIS, linewidth=0.8))
ax.annotate("16 轮", xy=(4999, grid[15, -1]), xytext=(5120, grid[15, -1] + 0.02),
            fontsize=8.8, color=INK, ha="left", va="bottom")
ax.set_xlim(300, 5320)
ax.set_ylim(1.40, 2.95)
ax.set_yticks([1.5, 2.0, 2.5])
ax.set_xlabel("训练步数", fontsize=9.5)
ax.set_ylabel("验证 loss", fontsize=9.5)
ax.set_title("B  五个深度：3000 步是共同拐点", loc="left", fontsize=11.5,
             fontweight="semibold", pad=8)
ax.legend(loc="upper right", fontsize=8.5, ncol=3, handlelength=1.3,
          columnspacing=1.0, borderaxespad=0.2, labelspacing=0.35)
style_axis(ax)

fig.text(0.055, 0.035, "数据：loop_train_metrics.csv；热力图只画 step 500 之后，"
                       "避免初始 4.33 压缩色标。",
         fontsize=8.2, color=GRAY, ha="left", va="bottom")
fig.savefig("loop_training_analysis.png", dpi=200)
plt.close(fig)
print("Saved loop_training_analysis.png")
