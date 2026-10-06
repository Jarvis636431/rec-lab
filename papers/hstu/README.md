# HSTU 阅读笔记

- 原文：Jiaqi Zhai 等，*Actions Speak Louder than Words: Trillion-Parameter Sequential Transducers for Generative Recommendations*，2024。[arXiv:2402.17152](https://arxiv.org/abs/2402.17152)；本地 [PDF](Zhai_2024_HSTU.pdf)，26 页，v3。
- 核心问题：大规模推荐行为序列具有高基数、异质性和持续变化的特点；论文重新审视把通用 Transformer 直接用于推荐的设计，并提出 HSTU 架构。
- 建议先读任务如何被表述为序列转导，再看 HSTU 相比普通注意力模块改变了什么，最后看质量、训练效率和规模化实验分别使用什么数据与硬件条件。
- 与仓库关系：路线图计划在 SASRec 之后做缩小版对照。现阶段没有 HSTU 实现；未来的小模型实验只能验证部分结构和成本，不等于复现论文中的超大规模系统或线上收益。
- 自检：HSTU 针对的是推荐行为序列的什么特点？若与 SASRec 对照，必须固定哪些数据与评估条件？
