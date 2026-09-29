# 01 · Logistic Regression CTR Baseline

这个实验完整跑通一次最小 CTR 预估链路：

```text
模拟曝光日志
→ 按生成顺序切分（模拟 IID 数据）
→ 仅使用训练集统计量标准化
→ LR 输出 logits
→ BCEWithLogitsLoss
→ 反向传播与 SGD
→ 按 Validation BCE 保存并恢复最佳模型
→ AUC / LogLoss / 校准误差
→ 检查模型权重
```

## 模型结构

```text
x → z = wᵀx + b → sigmoid(z) → pCTR
```

`model.py` 只返回 logits。训练时由 `BCEWithLogitsLoss` 稳定地组合 Sigmoid 与二元交叉熵；评估时才显式调用 Sigmoid 获得概率。

数据按 70% / 15% / 15% 顺序切分。生成器没有时间戳或时间漂移，因此这是模拟 IID 数据的顺序切分；真实日志需要按实际事件时间建立验证协议。

## 模拟数据规律

数据生成器人为设置了以下规律：

| 特征 | 对点击的影响 |
| --- | --- |
| 用户喜欢运动 | 正向 |
| 商品属于运动类别 | 正向 |
| 用户兴趣与商品类别匹配 | 强正向 |
| 商品价格更高 | 负向 |
| 展示位置更靠后 | 负向 |
| 晚间展示 | 弱正向 |

其中“兴趣匹配”是：

```text
user_sports × item_sports
```

第一次实验不把它提供给 LR，第二次通过 `--include-cross` 加入。两次结果的差异用于说明：LR 能学习输入的一阶权重，但不会自动创造乘法交叉特征。

## 运行

基础 LR：

```bash
uv run python -m models.lr.train
```

加入人工交叉特征：

```bash
uv run python -m models.lr.train --include-cross
```

常用参数：

```bash
uv run python -m models.lr.train \
  --samples 20000 \
  --epochs 40 \
  --batch-size 256 \
  --learning-rate 0.05 \
  --seed 42
```

低 CTR 与准确率陷阱实验：

```bash
uv run python -m models.lr.train --base-ctr 0.005
```

`--base-ctr` 表示其他特征产生影响前，参考曝光的基础点击概率，不是数据最终的整体 CTR。省略该参数时继续使用原始截距 `-2.6`，数据生成规律与原来的运行命令保持不变。训练输出会同时报告模型指标与 `always-negative baseline`，用于观察低 CTR 下“全部预测为不点击”为什么能获得很高 Accuracy，却具有 0 Recall 和 0.5 AUC（测试集包含两类时）。

配置了基础 CTR 的实验位于独立实验组，例如 `--base-ctr 0.005` 对应 `outputs/lr/baseline_base_ctr_0p005/`；每次运行在组内新建目录。

本次低 CTR 实验得到：

| 指标 | LR | 永远预测不点击 |
| --- | ---: | ---: |
| Accuracy | 0.992000 | 0.992000 |
| Recall | 0 | 0 |
| AUC | 0.782622 | 0.500000 |
| AP（字段名 `pr_auc`） | 0.047474 | 0.008000 |

两者 Accuracy 相同，但 LR 已经学到有效排序。详细解释见 [类别不平衡与正则化补充笔记](IMBALANCE_AND_REGULARIZATION.md)。

该实验测试集 3000 条、正样本仅 24 个，实际 CTR 0.8%；模型 predicted CTR 为 1.5035%，LogLoss 为 0.045618，ECE 为 0.007035。排序能力与校准必须分别检查，单个 seed 的指标不能作为稳定收益结论。

其中类别不平衡、指标选择、负采样和校准属于可复用于后续模型的搜广推基础；人工交叉特征和线性权重解释则是 LR 相关结论。补充笔记中单独整理了两类知识的边界。

默认跑完指定 Epoch，然后恢复 **Validation BCE 最低** 的模型。可以开启提前停止：

```bash
uv run python -m models.lr.train --epochs 200 --patience 5
```

`--patience 5` 表示验证损失连续 5 轮没有下降就停止；`--patience 0`（默认）关闭提前停止。两种情况下都会恢复最佳模型，之后才预测测试集；测试集不参与模型选择。

每次运行创建独立目录，包含 seed、UTC 时间和随机后缀；相同参数重复运行也不会覆盖。实验组路径保留，旧目录中的产物不会被覆盖：

```text
outputs/lr/baseline/seed_42_<UTC时间>_<后缀>/
outputs/lr/with_cross/seed_42_<UTC时间>_<后缀>/
outputs/lr/baseline_base_ctr_0p005/seed_42_<UTC时间>_<后缀>/
```

每次实验会生成：

- `config.json`：全部命令行参数、特征名、切分样本数/正负样本数/CTR、指标定义、Python/依赖版本和 Git commit/未提交变更状态；
- `history.json`：每轮 `epoch`、`train_loss`、`val_loss` 与 `is_best`，用于绘制学习曲线；Train Loss 是该轮各 Batch 更新前的 BCE 加权平均；
- `metrics.json`：最终测试指标、最佳轮次、实际轮数、提前停止状态和切分统计；
- `weights.json`：标准化特征空间中的权重与偏置；
- `model.pt`：最佳模型参数、特征名、标准化统计量、配置和指标。

`config.json` 中的 Git commit 与 dirty 状态用于追踪代码；dirty 为 true 时，需要同时保留未提交的代码修改，不能仅凭 commit 恢复该次实验。生成时间和目录后缀是运行标识，不影响数据/模型随机种子。

### 指标边界

`pr_auc` 沿用原有字段名，实际计算为 **Average Precision（AP）**，不是梯形积分的 PR 曲线面积。Precision、Recall、Accuracy 使用阈值 0.5；低 CTR 下模型可能有排序能力但在该阈值下没有正预测。

当测试集仅含一种标签时，AUC 不可定义；本项目同时把 AUC/AP 记为 JSON `null`，明确该集合不用于两类排序比较，不人为填成 0.5。LogLoss（显式指定两类）、ECE 和分类指标仍正常输出。输出每个集合的正样本数，方便识别低 CTR 下样本不足的情况。

ECE 使用 10 个等宽概率分箱；在极低 CTR 下多数预测可能落在第一个箱子，需结合 predicted/observed CTR 解读，不能只看 ECE。

### 重载最佳模型

新 checkpoint 的标准化统计量保存为 Tensor，可以使用 `weights_only=True` 重载：

```python
import torch
from models.lr.model import LogisticRegression

checkpoint = torch.load("<本次运行目录>/model.pt", map_location="cpu", weights_only=True)
model = LogisticRegression(len(checkpoint["feature_names"]))
model.load_state_dict(checkpoint["state_dict"])
model.eval()

# raw_features 是 float32 Tensor，列顺序严格对应 feature_names。
# price 列已执行 log1p；sports_match（如有）已完成 user_sports * item_sports。
# 复用训练时统计量，不在推理数据上重新拟合。
with torch.no_grad():
    features = (raw_features - checkpoint["mean"]) / checkpoint["std"]
    probabilities = torch.sigmoid(model(features))
```

## 实验结果

以下结果于 2026-09-28 重跑，使用默认配置：20,000 条样本、最多 40 个 Epoch、Batch Size 256、学习率 0.05、随机种子 42、patience 0。基础 LR 最佳轮为 39，加入交叉特征的最佳轮为 40。旧版保存最终轮，基础 LR 的 AUC 为 0.738938；现在按验证集选模，数字略有变化。

| 指标 | 基础 LR | 加入交叉特征 | 变化 |
| --- | ---: | ---: | ---: |
| AUC | 0.738985 | 0.757314 | +0.018329 |
| AP | 0.342444 | 0.348976 | +0.006532 |
| LogLoss | 0.291989 | 0.285596 | -0.006393 |
| ECE | 0.014614 | 0.008108 | -0.006506 |
| 预测 CTR | 10.5478% | 10.5094% | 更接近实际 CTR |
| 实际 CTR | 10.3667% | 10.3667% | 不变 |

基础 LR 的 Validation Loss 从 0.455192 降至最佳 0.287809；加入交叉特征后从 0.420732 降至 0.281647。两组训练均正常收敛，Train Loss 与 Validation Loss 接近，没有明显过拟合。

### 权重对比

| 特征 | 基础 LR | 加入交叉特征 |
| --- | ---: | ---: |
| `user_sports` | +0.575640 | +0.202694 |
| `item_sports` | +0.540150 | +0.118882 |
| `price` | -0.440441 | -0.450973 |
| `position` | -0.291012 | -0.305797 |
| `is_evening` | +0.162576 | +0.165271 |
| `sports_match` | — | +0.518141 |

未提供交叉特征时，LR 只能把匹配带来的部分提升分摊给 `user_sports` 和 `item_sports`。加入 `sports_match` 后，两个一阶权重下降，匹配关系被单独建模；与此同时，AUC、LogLoss 和 ECE 均有改善。

这些权重对应标准化后的输入空间，适合比较符号和实验间的变化，不应直接与数据生成器里的原始系数逐项比较。

## 实验结论

1. LR 能学习已有特征的一阶权重，但不会自动创造乘法交叉特征。
2. 人工加入 `user_sports × item_sports` 后，模型的排序能力和概率质量同时提升。
3. 价格、位置和晚间特征的权重方向符合数据生成规律。
4. 这次对照实验说明了传统 LR 对人工特征工程的依赖，并为 FM 自动学习二阶特征交互提供了动机。

## 实验需要回答的问题

1. Train Loss 和 Validation Loss 是否下降？
2. 测试集 AUC 是否明显高于 0.5？
3. 价格和展示位置的权重是否为负？
4. 晚间展示的权重是否为正？
5. 加入人工交叉特征后，AUC 和 LogLoss 是否改善？
6. 为什么 LR 无法在没有该特征时自己学出乘法关系？

## 下一步

工程收尾已完成。逐段阅读训练循环，把 Batch、logits、BCE、反向传播、更新、验证集选模与实际输出对应起来；下一轮进入 FM，与基础 LR 和人工交叉 LR 做同数据对照。负采样修正、L1/L2 与噪声特征实验按 [ROADMAP](../../ROADMAP.md) 逐步安排，不作为开始 FM 的前置阻塞项。[补充笔记](IMBALANCE_AND_REGULARIZATION.md) 中的准确率陷阱与 Early Stopping 已实现，采样与正则化实验尚待实现。
