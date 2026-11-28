"""Plot every available training and loop-depth metric with one command.

Run ``python plot.py`` from any directory. Figures are saved beside this file.
"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

ROOT = Path(__file__).resolve().parent
BLUE, ORANGE, RED, GRAY = "#2468b4", "#d48a28", "#c94b54", "#68727d"
QUARTILES = ("#91bce3", "#5d96cc", "#3273ac", "#174978")

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10,
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#b9c2cc", "axes.labelcolor": "#333c45",
    "text.color": "#202a34", "grid.color": "#e4e9ee",
    "savefig.dpi": 180,
})


def rows(filename):
    with (ROOT / filename).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def series(data, key):
    return np.array([float(row[key]) if row.get(key) else np.nan for row in data])


def finish(fig, filename):
    fig.savefig(ROOT / filename, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {ROOT / filename}")


def style(ax, xlabel, ylabel):
    ax.set(xlabel=xlabel, ylabel=ylabel)
    ax.grid(axis="y", linewidth=.8)
    ax.set_axisbelow(True)


def main():
    depth_rows = sorted(rows("loop_token_metrics.csv"), key=lambda r: int(r["n_loops"]))
    depth = series(depth_rows, "n_loops").astype(int)
    loss = series(depth_rows, "val_loss")
    accuracy = series(depth_rows, "accuracy") * 100
    best = int(np.nanargmin(loss))
    best_acc = int(np.nanargmax(accuracy))
    trained = int(depth[-1])
    summary = rows("loop_val_losses.csv")
    baseline = next(float(r["val_loss"]) for r in summary if r["n_loops"] == "gpt")
    history = rows("train_history.csv")
    models = {name: sorted((r for r in history if r["model"] == name),
                           key=lambda r: int(r["step"]))
              for name in {r["model"] for r in history}}

    # Figure 1: the main comparison, including the training trajectories.
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.7), layout="constrained")
    fig.suptitle("Validation loss | compute depth and training", fontsize=15, weight="bold")
    ax = axes[0]
    ax.plot(depth, loss, "o-", color=BLUE, lw=2, ms=4, label="Shared block")
    ax.axhline(baseline, color=ORANGE, ls="--", lw=1.8, label=f"Standard GPT: {baseline:.3f}")
    ax.scatter(depth[best], loss[best], s=90, color=BLUE, edgecolor="white", zorder=5)
    ax.annotate(f"Minimum: {depth[best]} loops, {loss[best]:.3f}",
                (depth[best], loss[best]), xytext=(-125, 28), textcoords="offset points",
                arrowprops={"arrowstyle": "-", "color": GRAY})
    ax.annotate(f"Trained depth: {trained}, {loss[-1]:.3f}",
                (trained, loss[-1]), xytext=(-128, 52), textcoords="offset points",
                arrowprops={"arrowstyle": "-", "color": GRAY})
    ax.set_xticks(depth[::max(1, len(depth)//8)])
    ax.set_xlim(depth[0]-.5, trained+1.5)
    ax.set_title("Final checkpoint: inference loops", loc="left")
    style(ax, "Number of shared-block passes", "Full validation loss")
    ax.legend(loc="upper right", frameon=False)

    ax = axes[1]
    for name, color, label in (("gpt", ORANGE, "Standard GPT"),
                               ("looped_gpt", BLUE, "Shared-block GPT")):
        if name not in models:
            continue
        data = models[name]
        steps = series(data, "step")
        ax.plot(steps, series(data, "val_loss"), "o-", lw=2, ms=3,
                color=color, label=f"{label} validation")
        ax.plot(steps, series(data, "train_loss"), ls=":", lw=1.5,
                color=color, alpha=.75, label=f"{label} train")
    ax.set_title("Training: solid = validation, dotted = train", loc="left")
    style(ax, "Training step", "Loss")
    ax.legend(fontsize=8, frameon=False)
    finish(fig, "looped_val_loss_vs_loops.png")

    # Figure 2: token-level behavior at the selected final checkpoint.
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9), layout="constrained")
    fig.suptitle("Final checkpoint | what each additional loop changes", fontsize=15, weight="bold")
    ax = axes[0, 0]
    ax.plot(depth, loss, "o-", color=BLUE, ms=3)
    ax.axhline(baseline, color=ORANGE, ls="--", label=f"Standard GPT {baseline:.3f}")
    ax.scatter(depth[best], loss[best], color=BLUE, s=70, zorder=5)
    ax.set_title(f"A  Lowest loss at {depth[best]} loops ({loss[best]:.3f})", loc="left")
    style(ax, "Loops", "Validation loss")
    ax.legend(frameon=False)

    ax = axes[0, 1]
    ax.plot(depth, accuracy, "o-", color=BLUE, ms=3)
    ax.scatter(depth[best_acc], accuracy[best_acc], color=BLUE, s=70, zorder=5)
    ax.set_title(f"B  Highest accuracy at {depth[best_acc]} loops ({accuracy[best_acc]:.1f}%)", loc="left")
    style(ax, "Loops", "Next-token top-1 accuracy (%)")

    ax = axes[1, 0]
    # Counts are fractions of *all* tokens, so their difference equals the accuracy gain.
    corrected = series(depth_rows, "became_correct") * 100
    broken = series(depth_rows, "became_wrong") * 100
    ax.plot(depth[1:], corrected[1:], "o-", color=BLUE, ms=3, label="Wrong to right")
    ax.plot(depth[1:], broken[1:], "o-", color=RED, ms=3, label="Right to wrong")
    ax.fill_between(depth[1:], broken[1:], corrected[1:], color=BLUE, alpha=.10)
    ax.set_title("C  Prediction flips per added loop", loc="left")
    style(ax, "New loop number", "Share of all tokens (%)")
    ax.legend(frameon=False)

    ax = axes[1, 1]
    for i, color in enumerate(QUARTILES, 1):
        values = series(depth_rows, f"initial_loss_q{i}")
        minimum = int(np.nanargmin(values))
        ax.plot(depth, values, color=color, lw=2, label=f"Q{i}: best at {depth[minimum]}")
        ax.scatter(depth[minimum], values[minimum], color=color, s=35, zorder=4)
    ax.set_title("D  Fixed groups by initial token difficulty", loc="left")
    style(ax, "Loops", "Mean token loss")
    ax.legend(fontsize=8, frameon=False)
    fig.text(.5, -.005, "Q1 = easiest at loop 1; Q4 = hardest. Groups stay fixed across loops.",
             ha="center", color=GRAY, fontsize=9)
    finish(fig, "loop_depth_analysis.png")

    # Figure 3: fixed-sample depth measurements over training.
    train = rows("loop_train_metrics.csv")
    steps = sorted({int(r["step"]) for r in train if int(r["step"]) > 0})
    step_index = {step: i for i, step in enumerate(steps)}
    depth_index = {int(d): i for i, d in enumerate(depth)}
    grid = np.full((len(depth), len(steps)), np.nan)
    for row in train:
        step, d = int(row["step"]), int(row["n_loops"])
        if step in step_index and d in depth_index:
            grid[depth_index[d], step_index[step]] = float(row["val_loss"])
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2), layout="constrained")
    fig.suptitle("Training checkpoints | fixed validation sample", fontsize=15, weight="bold")
    ax = axes[0]
    cmap = LinearSegmentedColormap.from_list("loss", ["#f0f6fc", "#85b5df", "#174978"])
    image = ax.imshow(grid, origin="lower", aspect="auto", cmap=cmap)
    if np.isfinite(grid).all():
        winners = depth[np.argmin(grid, axis=0)]
        ax.plot(np.arange(len(steps)), [depth_index[int(d)] for d in winners],
                "o-", color="white", markeredgecolor="#202a34", ms=4, lw=1.5,
                label="Best depth")
        ax.legend(loc="upper right", fontsize=8)
    ax.set_xticks(np.arange(len(steps)), [str(s) for s in steps], rotation=45, ha="right")
    ax.set_yticks(np.arange(len(depth)), [str(d) for d in depth])
    ax.set(xlabel="Training step", ylabel="Loops", title="A  Validation loss by depth")
    fig.colorbar(image, ax=ax, label="Validation loss")

    ax = axes[1]
    selected = sorted(set([int(depth[0]), *[int(depth[min(i, len(depth)-1)])
                                          for i in (3, 7, 11)], trained]))
    colors = plt.cm.Blues(np.linspace(.4, .95, len(selected)))
    for d, color in zip(selected, colors):
        ax.plot(steps, grid[depth_index[d]], "o-", color=color, ms=3,
                lw=1.8, label=f"{d} loops")
    ax.set_title("B  Selected depths during training", loc="left")
    style(ax, "Training step", "Fixed-sample validation loss")
    ax.legend(frameon=False, fontsize=8)
    finish(fig, "loop_training_analysis.png")
    print(f"Best full-validation loss: {loss[best]:.4f} at {depth[best]} loops; "
          f"standard GPT: {baseline:.4f}; best accuracy: {accuracy[best_acc]:.2f}% "
          f"at {depth[best_acc]} loops.")


if __name__ == "__main__":
    main()
