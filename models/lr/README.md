# 01 · Logistic Regression CTR Baseline

这个实验完整跑通一次最小 CTR 预估链路：

```text
模拟曝光日志
→ 按时间切分
→ 仅使用训练集统计量标准化
→ LR 输出 logits
→ BCEWithLogitsLoss
→ 反向传播与 SGD
→ AUC / LogLoss / 校准误差
→ 检查模型权重
```

## 模型结构

```text
x → z = wᵀx + b → sigmoid(z) → pCTR
```

`model.py` 只返回 logits。训练时由 `BCEWithLogitsLoss` 稳定地组合 Sigmoid 与二元交叉熵；评估时才显式调用 Sigmoid 获得概率。

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

结果保存在：

```text
outputs/lr/baseline/
outputs/lr/with_cross/
```

每次实验会生成：

- `metrics.json`：测试集 AUC、LogLoss、ECE 与基础信息；
- `weights.json`：标准化特征空间中的权重与偏置；
- `model.pt`：模型参数、特征名和标准化统计量。

## 实验结果

以下结果使用默认配置：20,000 条样本、40 个 Epoch、Batch Size 256、学习率 0.05、随机种子 42。

| 指标 | 基础 LR | 加入交叉特征 | 变化 |
| --- | ---: | ---: | ---: |
| AUC | 0.738938 | 0.757314 | +0.018376 |
| LogLoss | 0.292002 | 0.285596 | -0.006406 |
| ECE | 0.014450 | 0.008108 | -0.006342 |
| 预测 CTR | 10.5475% | 10.5094% | 更接近实际 CTR |
| 实际 CTR | 10.3667% | 10.3667% | 不变 |

基础 LR 的 Validation Loss 从 0.455192 降至 0.287810；加入交叉特征后从 0.420732 降至 0.281647。两组训练均正常收敛，Train Loss 与 Validation Loss 接近，没有明显过拟合。

### 权重对比

| 特征 | 基础 LR | 加入交叉特征 |
| --- | ---: | ---: |
| `user_sports` | +0.576785 | +0.202694 |
| `item_sports` | +0.541536 | +0.118882 |
| `price` | -0.440745 | -0.450973 |
| `position` | -0.290694 | -0.305797 |
| `is_evening` | +0.162284 | +0.165271 |
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

逐段阅读训练循环，把 Batch、logits、BCE、反向传播、参数更新和评估指标与实际输出对应起来；随后进入 FM，让模型通过隐向量自动学习二阶特征交互，并与这组 LR 结果进行对照。
