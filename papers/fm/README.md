# 《Factorization Machines》阅读笔记

## 原文与出处

- 作者：Steffen Rendle。
- 发表：2010 IEEE International Conference on Data Mining（ICDM），第 995–1000 页。
- DOI：[10.1109/ICDM.2010.127](https://doi.org/10.1109/ICDM.2010.127)。
- 本地原文：[Rendle_2010_Factorization_Machines.pdf](Rendle_2010_Factorization_Machines.pdf)，6 页。
- PDF 获取自[公开论文归档镜像](https://github.com/wzhe06/Ad-papers/blob/master/Factorization%20Machines/Factorization%20Machines%20Rendle2010.pdf)；论文题目、作者和页数已与原文及[书目信息](https://dblp.org/rec/conf/icdm/Rendle10.html)核对。SHA-256：`ba9172e0f3b4e597475a4ae0817633bbee4823bf34850f822ccf481fb4009d49`。

## 论文要解决什么问题

推荐数据通常很稀疏：一个样本只激活少量用户、物品和上下文特征。若为每一对特征单独估计一个交互系数，很多组合几乎没有训练样本，参数难以可靠学习。论文用每个特征的隐向量点积表示交互系数，让不同特征对共享参数，并证明二阶 FM 可在线性时间内计算。

## 核心公式

二阶 FM 的预测分数为：

```text
y_hat(x) = w0 + Σ_i w_i x_i + Σ_{i<j} <v_i, v_j> x_i x_j
```

其中 `w0` 是偏置，`w_i` 是一阶权重，`v_i` 是第 `i` 个特征的 `k` 维隐向量。逐对计算二阶项需要 `O(n²k)`；代数变形后：

```text
Σ_{i<j} <v_i, v_j> x_i x_j
  = 1/2 Σ_f [(Σ_i v_if x_i)² - Σ_i v_if² x_i²]
```

计算量降到 `O(nk)`；对稀疏输入，还可以只遍历非零特征。公式证明与计算复杂度见论文第 III 节。一个关键理解是：FM **仍然只显式建模到二阶**，其优势不是自动获得任意高阶关系，而是在稀疏数据中共享交互参数。

## 与仓库实验的对应和边界

| 论文概念 | 仓库对应 |
| --- | --- |
| 一阶项和隐向量点积二阶项 | [`models/fm/model.py`](../../models/fm/model.py) |
| 二阶项的线性复杂度公式 | `FactorizationMachine.forward()` |
| 与逐对公式等价 | [`models/fm/tests/test_model.py`](../../models/fm/tests/test_model.py) |
| 稀疏输入下的参数共享动机 | 当前只在 5 个稠密特征上做原理演示，尚未验证 |

本仓库的 FM 用模拟 CTR 标签、Sigmoid/BCE 和 Adam 训练；论文讨论的是适用于多种预测任务的通用 FM，并重点分析稀疏特征。我们在三个 seed 上看到 FM 无需人工 `sports_match` 即接近人工交叉 LR，详情见[实验记录](../../models/fm/README.md)。这不是论文数据集或论文指标的复现。

## 建议阅读顺序与自检

1. 先读第 II 节的稀疏输入例子：一个用户–物品样本有哪些非零特征？
2. 重点读第 III 节的模型公式、交互参数为何可共享，以及复杂度变形。
3. 选读第 IV–V 节：FM 与多项式 SVM、矩阵分解的关系。
4. 对照仓库模型代码，手算只有两个非零特征时的二阶项；再运行 `uv run pytest -q models/fm/tests/test_model.py`。

读完应能回答：为什么 LR 无法表达未提供的乘法交互？为什么隐向量点积比为每对特征单独设一个系数更适合稀疏数据？为什么本仓库目前还没有验证论文最重要的稀疏场景优势？
