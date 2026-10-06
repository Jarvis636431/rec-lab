# 论文阅读目录

每篇论文单独放在一个子目录：原文 PDF 与 `README.md` 阅读笔记相邻。笔记记录出处、核心问题、建议阅读顺序、与本仓库实验的对应关系，以及尚未验证的结论。

| 学习阶段 | 论文 | 阅读笔记 | 仓库状态 |
| --- | --- | --- | --- |
| 二阶特征交互 | Rendle (2010), *Factorization Machines* | [FM](fm/README.md) | [FM 实验已完成](../models/fm/README.md) |
| 低阶与深层交互 | Guo et al. (2017), *DeepFM* | [DeepFM](deepfm/README.md) | 待实现 |
| 候选相关兴趣 | Zhou et al. (2018), *Deep Interest Network* | [DIN](din/README.md) | 待实现 |
| 自注意力序列推荐 | Kang & McAuley (2018), *SASRec* | [SASRec](sasrec/README.md) | 待实现 |
| 生成式推荐序列 | Zhai et al. (2024), *HSTU* | [HSTU](hstu/README.md) | 待实现 |
| 语义 ID 生成式检索 | Rajput et al. (2023), *TIGER* | [TIGER](tiger/README.md) | 待实现 |
| 统一召回与排序 | Deng et al. (2025), *OneRec* | [OneRec](onerec/README.md) | 阅读阶段 |

表格按学习依赖排列，HSTU 与 TIGER 可并行理解；OneRec 是较新的工业方案，并非历史意义上的“经典”基线。后续读到新论文时再增加目录。论文原文与本仓库实验结论分开记录，避免把简化实现当作论文复现。
