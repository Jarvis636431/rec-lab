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
python -m models.lr.train
```

加入人工交叉特征：

```bash
python -m models.lr.train --include-cross
```

常用参数：

```bash
python -m models.lr.train \
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

## 实验需要回答的问题

1. Train Loss 和 Validation Loss 是否下降？
2. 测试集 AUC 是否明显高于 0.5？
3. 价格和展示位置的权重是否为负？
4. 晚间展示的权重是否为正？
5. 加入人工交叉特征后，AUC 和 LogLoss 是否改善？
6. 为什么 LR 无法在没有该特征时自己学出乘法关系？

## 下一步

完成结果解读后，学习 AUC、LogLoss 和概率校准，再进入 FM。FM 将尝试通过隐向量自动学习二阶特征交互。
