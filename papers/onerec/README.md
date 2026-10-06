# OneRec 阅读笔记

- 原文：Jiaxin Deng 等，*OneRec: Unifying Retrieve and Rank with Generative Recommender and Preference Alignment*，2025。[arXiv:2502.18965](https://arxiv.org/abs/2502.18965)；本地 [PDF](Deng_2025_OneRec.pdf)，10 页，v1。arXiv 条目标题使用了 “Iterative Preference Alignment”，本地 PDF 首页标题省略了 “Iterative”。
- 核心问题：生成式推荐若只替代召回阶段，后续仍依赖单独排序器。OneRec 尝试用统一生成模型处理召回和排序，并进一步对齐用户偏好。
- 结构主线：编码用户历史，逐步生成推荐视频；采用 session-wise 生成以考虑一次请求中的多个结果；使用稀疏 MoE 扩展容量，并通过偏好对齐模块优化生成结果。论文中的工业指标是作者报告，仓库尚未复现。
- 建议先读系统输入输出和 session-wise 目标，再读召回/排序统一的训练与推理路径，最后读偏好对齐、奖励模型和实验消融。不要把论文的工业部署结果当作本地可直接达到的目标。
- 与仓库关系：目前适合做阅读与架构对照。等双塔、SASRec、TIGER 等较小闭环完成后，再决定是否实现局部概念验证；当前没有 OneRec 代码或线上数据。
- 自检：一次请求中生成多个结果与逐个预测下一物品有什么区别？偏好对齐的监督信号从哪里来？
