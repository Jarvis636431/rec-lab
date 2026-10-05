# LR 训练循环逐步验收

这份笔记对应 [`train.py`](train.py) 的默认实验：20,000 条模拟曝光、5 个输入特征、Batch Size 256、学习率 0.05、seed 42。目标是能从一个 Batch 一直解释到最佳模型的选择和最终测试。完整实验结果见 [README](README.md)。

## 1. 数据进入训练循环前

[`prepare_data`](data.py) 先生成曝光与点击标签，再按生成顺序分成训练 70%、验证 15%、测试 15%。默认得到 `train_x: (14000, 5)`、`train_y: (14000,)`、`val_x: (3000, 5)` 和 `test_x: (3000, 5)`。这里的顺序切分针对模拟 IID 数据；生成器没有真实时间戳或时间漂移。

5 列依次是 `user_sports`、`item_sports`、`price`、`position`、`is_evening`。`price` 在生成时已做 `log1p`。均值和标准差只从训练集计算，然后用于三个集合，避免把验证或测试数据的统计量带入训练。加上 `--include-cross` 时才有第 6 列 `sports_match`。

[`train`](train.py) 先调用 `set_seed`，建立打乱顺序的训练 DataLoader，然后创建 LR、`BCEWithLogitsLoss` 和 SGD。随机种子同时影响模型初始化与 Batch 顺序。

## 2. 一个 Batch 中发生什么

默认配置下，一个完整 Batch 的形状如下。最后不足 256 条的 Batch 会更小。

| 张量或参数 | 形状 | 含义 |
| --- | --- | --- |
| `batch_x` | `(256, 5)` | 256 次曝光的 5 个标准化特征 |
| `batch_y` | `(256,)` | 每次曝光的 0/1 点击标签 |
| `model.linear.weight` | `(1, 5)` | 每列特征的线性权重 |
| `model.linear.bias` | `(1,)` | 截距 |
| `logits = model(batch_x)` | `(256,)` | 每次曝光的原始分数，尚不是概率 |
| `loss` | `()` | 该 Batch 的平均 BCE，标量 |

模型计算 `logit_i = w · x_i + b`。[`model.py`](model.py) 用 `squeeze(-1)` 去掉输出的最后一维。训练时直接把 logits 交给 `BCEWithLogitsLoss`；它在数值稳定的实现中结合 Sigmoid 与二元交叉熵。只有预测指标需要概率时，才显式计算 `sigmoid(logits)`。

训练循环的核心顺序是：

```python
optimizer.zero_grad()             # 清除上一个 Batch 留下的梯度
logits = model(batch_x)           # 前向传播，得到 (B,) 的 logits
loss = criterion(logits, batch_y) # 当前参数下的平均 BCE
loss.backward()                   # 计算 loss 对权重和偏置的梯度
optimizer.step()                  # SGD 按梯度更新参数
```

`zero_grad()` 必须在本次 `backward()` 前执行，否则 PyTorch 默认会累加梯度。`backward()` 只计算并存放梯度；真正修改参数的是 `step()`。当前没有正则项，SGD 更新可写为 `w_new = w_old - 0.05 × grad_w`，偏置同理。

### 固定 seed 的第一批数值

用仓库当前环境运行以下代码，可单独复现一个 Batch。创建模型的顺序与训练脚本一致，然后才从 DataLoader 取第一批。

```python
import torch
from torch import nn

from models.lr.data import prepare_data
from models.lr.model import LogisticRegression
from models.lr.train import make_loader, set_seed

set_seed(42)
data = prepare_data(samples=20_000, seed=42)
loader = make_loader(data.train_x, data.train_y, batch_size=256, shuffle=True)
model = LogisticRegression(len(data.feature_names))
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.05)

batch_x, batch_y = next(iter(loader))
optimizer.zero_grad()
logits = model(batch_x)
loss = criterion(logits, batch_y)
old_weight = model.linear.weight.detach().clone()
old_bias = model.linear.bias.detach().clone()
loss.backward()

print(batch_x.shape, batch_y.shape, logits.shape)
print(float(logits[0]), float(torch.sigmoid(logits[0])), float(batch_y[0]))
print(float(loss), model.linear.weight.grad, model.linear.bias.grad)
optimizer.step()
print(old_weight, model.linear.weight.detach())
print(old_bias, model.linear.bias.detach())
```

本地运行中，第一批 `loss ≈ 0.774713`；第一个样本的 `logit ≈ -0.541274`，Sigmoid 后概率约为 `0.367891`，标签为 `0`。第一项权重从 `0.341912` 变为 `0.339899`，其梯度约为 `0.040252`：`0.341912 - 0.05 × 0.040252 ≈ 0.339899`。偏置从 `0.090244` 变为 `0.069013`，梯度约为 `0.424616`。不同 PyTorch 版本或平台可能产生不同的具体数值，形状、计算顺序和更新关系才是验收重点。

## 3. 从 Batch 到每轮损失

一个 Epoch 遍历全部 14,000 条训练样本。代码将每个 Batch 的平均 BCE 乘以该 Batch 的样本数，最后除以总样本数，得到 `train_loss`。由于参数在每个 Batch 后都更新，这个值是**各 Batch 更新前损失的加权平均**，不是 Epoch 结束时用最终参数重新计算的训练集损失。

每轮结束后，`predict` 以 `eval()` 和 `no_grad()` 在完整验证集上计算 `val_loss`。验证阶段不更新权重，记录的验证 BCE 对应这一轮结束时的同一组参数。两条曲线写入 `history.json`。

## 4. 为什么保存并恢复最佳轮

每轮把 `val_loss` 与迄今最小值比较；严格更小时，深拷贝该轮参数并记录 `best_epoch`。`--patience 0` 默认跑满指定轮数；设置正数后，验证损失连续指定轮数未改善便提前停止。无论是否提前停止，测试前都会恢复验证 BCE 最低那轮的参数。

测试集只在恢复最佳参数后预测一次。随后计算 AUC、AP（输出字段名为 `pr_auc`）、LogLoss、ECE 和阈值 0.5 下的分类指标。测试指标用于最终报告，不参与选轮或调参。`metrics.json` 记录最佳轮与测试指标；`model.pt` 保存恢复后的参数及标准化统计量，重载时必须保持相同特征顺序。

## 5. 自检问题

1. 为什么 `batch_x` 是 `(256, 5)`，而 `logits` 是 `(256,)`？
2. 为什么训练损失接收 logits，预测指标却接收 Sigmoid 后的概率？
3. `zero_grad()`、`backward()` 和 `step()` 分别改变了什么？能否用上面的第一项权重核对一次更新？
4. 为什么 `train_loss` 不能解释为轮末模型在整个训练集上的 BCE？
5. 哪个集合决定 `best_epoch`？测试集在何时被读取？
6. 为什么重载模型时还需要保存的 `feature_names`、`mean` 和 `std`？

能独立回答这六个问题，并把答案对应到 [`data.py`](data.py)、[`model.py`](model.py) 和 [`train.py`](train.py)，就完成了这一步的个人验收。
