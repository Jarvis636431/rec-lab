# Rec Lab

搜广推模型学习与实验仓库。每个模型拥有独立目录，保留清晰的模型结构、训练过程、评估结果与学习笔记。

## 学习路线

```text
LR → FM → 稀疏字段与 MLP 对照 → DeepFM → DCN-v2 → Two Tower → DIN → ESMM → SASRec
   → HSTU → TIGER / Semantic ID → 多模态与统一建模 → 腾讯竞赛演练
```

已完成模型：[`LR`](models/lr/README.md)、[`FM`](models/fm/README.md)、[`DeepFM`](models/deepfm/README.md)。FM 与基础 LR、人工交叉 LR 的同数据三 seed 对照见 [FM 实验记录](models/fm/README.md)。

多字段稀疏模拟数据上的 [`LR / FM / MLP 对照`](models/sparse_ctr/README.md)和 [`DeepFM 分支消融`](models/deepfm/README.md)已完成；当前合成数据上完整 DeepFM 未稳定超过仅 FM 分支。

完整学习与竞赛准备计划：[`ROADMAP.md`](ROADMAP.md)，暂按每周 10–12 小时推进到 2027 年 4 月的参赛能力验收。

LR 补充笔记：[`类别不平衡与正则化`](models/lr/IMBALANCE_AND_REGULARIZATION.md)

论文原文与阅读笔记：[`papers/`](papers/README.md)，涵盖 FM、DeepFM、DIN、SASRec、HSTU、TIGER 和 OneRec。

## 环境准备

项目使用 Python 3.9+、PyTorch 和 [uv](https://docs.astral.sh/uv/)：

```bash
uv sync --extra dev
```

`uv sync` 根据 `pyproject.toml` 和 `uv.lock` 创建可复现的 `.venv`；`--extra dev` 同时安装 pytest。后续命令通过 `uv run` 自动使用该环境，无需手动激活。

## 运行第一个实验

```bash
uv run python -m models.lr.train
uv run python -m models.lr.train --include-cross
uv run python -m models.fm.train --embedding-dim 16
uv run python -m models.sparse_ctr.train --model fm
uv run python -m models.deepfm.train --variant deepfm
```

第一次实验不提供人工交叉特征，第二次加入 `user_sports × item_sports`，用于观察 LR 为什么依赖人工特征工程。

第一个 FM 实验仍只接收原始五列特征，通过二阶隐向量学习交互。它与两组 LR 的三个 seed 对照见 [`models/fm/README.md`](models/fm/README.md)；多字段稀疏实验使用另一套模拟数据，结果不能与下表直接比较。

默认参数（20,000 条样本、最多 40 个 Epoch、随机种子 42；恢复验证集最佳模型）的结果：

| 实验             |      AUC |  LogLoss |      ECE |
| ---------------- | -------: | -------: | -------: |
| 基础 LR          | 0.738985 | 0.291989 | 0.014614 |
| 加入人工交叉特征 | 0.757314 | 0.285596 | 0.008108 |

加入匹配特征后，AUC 提升 0.018329，LogLoss 下降约 2.19%，ECE 下降约 44.5%。详细的实验设计、权重变化和结论见 [`models/lr/README.md`](models/lr/README.md)。

每次运行会在实验组下创建独立目录，保存配置与环境、每轮损失、最终指标和验证集最佳模型。使用 `--patience 5` 可开启提前停止；默认完成指定轮数后恢复最佳模型。基础测试命令：

```bash
uv run pytest -q
```

## 目录约定

```text
rec-lab/
├── models/       # 每个模型一个独立学习单元
├── outputs/      # 模型、指标与实验产物，不提交 Git
├── pyproject.toml
└── uv.lock       # uv 锁定的可复现依赖版本
```

前三个模型完成前不提前封装通用 Trainer，确保数据、前向传播、损失、反向传播和参数更新在代码中清晰可见。
