# 04 · DeepFM 与共享 embedding 消融

## 要回答的问题

[稀疏 CTR 对照](../sparse_ctr/README.md)已经给出 LR、FM、MLP 的独立基线。这里按 [DeepFM 原论文阅读笔记](../../papers/deepfm/README.md)实现“一阶项 + FM 二阶项 + 深层网络”结构，并检验两条分支组合是否比单独分支更好。论文的工业数据和结果没有在本仓库复现。

## 模型结构

[`model.py`](model.py) 对每条曝光取六个类别 token 和两个标准化数值字段。类别 token 进入**同一张** `feature_embeddings` 表；数值字段乘以可学习的 `numeric_embeddings`。得到的八个向量同时送入：

1. FM 分支：用 `½[(Σv)²−Σv²]` 计算所有活跃字段的两两交互；
2. 深层分支：将八个向量拼接，经 `64 → 32 → 1` MLP 得到额外 logit。

类别的一阶偏置、数值线性项和全局偏置始终存在。最终输出是各启用分支的 **logits 之和**，训练时交给 `BCEWithLogitsLoss`。测试验证 FM 的高效二阶项与逐对计算一致，且 FM 和深层分支都能向同一张 embedding 表回传梯度。

消融版本为：

| `--variant` | 输出 | 用途 |
| --- | --- | --- |
| `fm_only` | 一阶 + FM 二阶 | 同一代码结构中的 FM 对照 |
| `deep_only` | 一阶 + 深层分支 | 检查深层分支的独立贡献 |
| `deepfm` | 一阶 + FM 二阶 + 深层分支 | 完整 DeepFM，两个分支共享 embedding |

这里的 `deep_only` **含一阶项**，因此不等同于前一实验中只有 embedding + MLP 的 `SparseMLP`。`fm_only` 与前一实验的 `SparseFM` 在模型思想上对应，但本实验固定了独立于模型初始化的 Batch 顺序；两批运行结果不应期待逐位相同。三种版本都使用相同的原始数据、字段处理和 Batch 顺序，关闭的分支不创建额外参数。

## 数据、训练与复现

[`data.py`](data.py) 直接复用 `models.sparse_ctr.data.prepare_data`：默认 50,000 条模拟曝光，35,000/7,500/7,500 顺序切分；六个类别字段只用训练集建词表，两个数值字段只用训练集统计量标准化。不同 seed 生成不同的隐藏偏好和标签，因此优先比较同 seed 的配对差异。数据无真实时间漂移、用户历史或在线曝光策略。

三个版本固定 embedding 维度 8、Adam、学习率 0.01、Batch Size 512、最多 25 Epoch、`patience=5`；每轮按验证 BCE 选模，恢复最佳参数后评估测试集。学习率与已有稀疏基线保持一致，不根据本次测试结果调参。

```bash
uv run pytest -q
for seed in 7 42 2026; do
  for variant in fm_only deep_only deepfm; do
    uv run python -m models.deepfm.train --variant "$variant" --seed "$seed"
  done
done
```

[`train.py`](train.py) 为该实验保留显式训练循环。每次运行独立保存 `config.json`、`history.json`、`metrics.json`、`model.pt`；checkpoint 包含最佳参数、训练集词表和数值标准化统计量。运行产物位于 `outputs/deepfm/`，不提交 Git。

## 三个 seed 的测试结果

`AP` 是 Average Precision，代码字段仍名为 `pr_auc`。同一 seed 的三组使用完全相同的测试集。

| seed | 版本 | AUC ↑ | AP ↑ | LogLoss ↓ | ECE ↓ | 最佳轮 |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 7 | FM only | 0.747383 | 0.215641 | 0.220880 | 0.008265 | 1 |
| 7 | Deep only | 0.741031 | 0.217686 | 0.221435 | 0.009054 | 2 |
| 7 | DeepFM | 0.747288 | 0.220764 | 0.220222 | 0.010526 | 1 |
| 42 | FM only | 0.657678 | 0.151080 | 0.276312 | 0.003418 | 1 |
| 42 | Deep only | 0.649214 | 0.150103 | 0.278544 | 0.008606 | 2 |
| 42 | DeepFM | 0.651374 | 0.154023 | 0.276899 | 0.007652 | 1 |
| 2026 | FM only | 0.691785 | 0.130834 | 0.216771 | 0.003673 | 1 |
| 2026 | Deep only | 0.677656 | 0.120567 | 0.218713 | 0.002790 | 1 |
| 2026 | DeepFM | 0.690718 | 0.128635 | 0.216910 | 0.002835 | 1 |

| 版本 | AUC，均值 ± 样本标准差 | AP，均值 ± 样本标准差 | LogLoss，均值 ± 样本标准差 |
| --- | ---: | ---: | ---: |
| FM only | 0.698949 ± 0.045280 | 0.165852 ± 0.044291 | 0.237988 ± 0.033253 |
| Deep only | 0.689300 ± 0.047003 | 0.162785 ± 0.049787 | 0.239564 ± 0.033785 |
| DeepFM | 0.696460 ± 0.048214 | 0.167808 ± 0.047586 | 0.238010 ± 0.033719 |

完整 DeepFM 相比 `fm_only` 的配对 AUC 差值为 -0.000095、-0.006304、-0.001067；LogLoss 差值为 -0.000658、+0.000587、+0.000138。AP 在前两个 seed 略高、第三个略低。**这份合成数据上没有稳定的组合收益**；不能因为模型包含两个分支就预期指标一定提升。

三组的最佳轮大多在第 1–2 轮，之后验证损失上升。这提示长尾 ID 和有限样本下容易过拟合。seed 42 的 `fm_only` 有 9,226 个参数，`deep_only` 与完整 DeepFM 各有 15,499 个；更大的模型并未自动改善泛化。ECE 也没有稳定改善。下一轮若研究正则化、维度或样本量，应只通过验证集选择，并保留独立测试集最终报告。

### 本次运行路径

| seed | FM only | Deep only | DeepFM |
| ---: | --- | --- | --- |
| 7 | `outputs/deepfm/fm_only/seed_7_20261006T161042781907Z_14933cf2` | `outputs/deepfm/deep_only/seed_7_20261006T161042817017Z_c8842f89` | `outputs/deepfm/deepfm/seed_7_20261006T161042783985Z_34618b8d` |
| 42 | `outputs/deepfm/fm_only/seed_42_20261006T161029937447Z_d152e62c` | `outputs/deepfm/deep_only/seed_42_20261006T161029937772Z_a5dd05dd` | `outputs/deepfm/deepfm/seed_42_20261006T161029937687Z_6fa81a59` |
| 2026 | `outputs/deepfm/fm_only/seed_2026_20261006T161042781102Z_ff7041c9` | `outputs/deepfm/deep_only/seed_2026_20261006T161042780002Z_1f15ae8d` | `outputs/deepfm/deepfm/seed_2026_20261006T161042782561Z_3ffc2f0e` |

## 下一步

DeepFM 的结构和分支消融已经完成。按路线图，下一步是 DCN/DCN-v2 的显式交叉；也可在进入新结构前先针对本实验的快速过拟合，做一次受控的 L2 与噪声特征实验。
