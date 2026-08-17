# Rec Lab

搜广推模型学习与实验仓库。每个模型拥有独立目录，保留清晰的模型结构、训练过程、评估结果与学习笔记。

## 学习路线

```text
LR → FM → DeepFM → DCN → Two Tower → DIN → ESMM → SASRec
```

当前模型：[`models/lr`](models/lr/README.md)

## 环境准备

项目使用 Python 3.9+ 和 PyTorch：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

## 运行第一个实验

```bash
python -m models.lr.train
python -m models.lr.train --include-cross
```

第一次实验不提供人工交叉特征，第二次加入 `user_sports × item_sports`，用于观察 LR 为什么依赖人工特征工程。

## 目录约定

```text
rec-lab/
├── models/       # 每个模型一个独立学习单元
├── outputs/      # 模型、指标与实验产物，不提交 Git
└── pyproject.toml
```

前三个模型完成前不提前封装通用 Trainer，确保数据、前向传播、损失、反向传播和参数更新在代码中清晰可见。
