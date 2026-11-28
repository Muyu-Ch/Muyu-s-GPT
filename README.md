本人的transformer学习小仓库，做做微型小实验来学习
运行环境：搭载 NVIDIA GeForce RTX 5060 Laptop GPU 的笔记本，Ubuntu 26.04。

`python train.py` 只训练、保存模型检查点和验证指标；需要生成文本时单独运行 `python generate.py`。安装 `numpy` 和 `matplotlib` 后，运行 `python plot.py` 一次读取全部四份 CSV，生成 `looped_val_loss_vs_loops.png`、`loop_depth_analysis.png` 和 `loop_training_analysis.png`。绘图脚本可以从任意工作目录运行，输出保存在项目目录。

训练现在带 warmup + cosine 学习率、weight decay、梯度裁剪和早停；`gpt_checkpoint.pth` / `looped_gpt_checkpoint.pth` 保存的是验证 loss 最好的 checkpoint，而不是最后一步。

运行 `train.py` 后，除了 `train_history.csv`（每次评估的 train/val loss 和学习率，用来检查何时开始过拟合），循环模型还会额外生成两个监测文件：

- `loop_train_metrics.csv`：每次训练日志时，在固定的 8 批验证样本上记录 1 到 `n_loop` 轮的指标（`step` 是训练步数）。
- `loop_token_metrics.csv`：训练结束后，在完整验证批次上记录同样的指标。

`accuracy` 是下一字符 top-1 正确率；`loss_p10/p50/p90` 是逐 token 交叉熵分位数。`initial_loss_q1` 到 `q4` 按第 1 轮的 token loss 从低到高固定分组，观察不同难度的 token 随循环加深如何变化。相邻两轮中，单 token loss 降低超过 0.01 计为 `fraction_improved`，变化绝对值不超过 0.01 计为 `fraction_flat`，升高超过 0.01 计为 `fraction_worsened`。`became_correct` 和 `became_wrong` 分别是错变对、对变错的 token 占全部 token 的比例；`gain_prev_correct/wrong` 是上一轮预测对/错的 token 在本轮的平均 loss 降幅。
