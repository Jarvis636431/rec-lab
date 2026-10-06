# TIGER 阅读笔记

- 原文：Shashank Rajput 等，*Recommender Systems with Generative Retrieval*，2023。[arXiv:2305.05065](https://arxiv.org/abs/2305.05065)；本地 [PDF](Rajput_2023_TIGER.pdf)，17 页，v3。TIGER 是文中提出的生成式检索方法名。
- 核心问题：传统召回通常先得到用户向量，再从物品向量索引中近邻检索。论文改为让模型自回归生成目标物品的离散标识。
- 结构主线：物品内容表示 → 量化得到多级 Semantic ID → 用历史物品的 ID 序列预测下一物品的 ID → 将生成的 ID 映射回合法物品。Semantic ID 是物品编码，不能把任意生成的 token 序列都当成目录中的有效物品。
- 建议先读 Semantic ID 如何构建与去重，再读序列到 ID 的生成目标、解码及检索评估。特别注意训练数据、候选目录和冷启动物品的定义。
- 与仓库关系：未来需从内容向量到语义 ID、生成、合法物品映射形成小闭环，再与 SASRec/双塔在固定候选口径下比较。当前没有 TIGER 实现或内容数据。
- 自检：Semantic ID 与随机物品 ID 的区别是什么？ID 碰撞、非法 ID 和解码延迟如何影响系统结果？
