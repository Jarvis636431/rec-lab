# DeepFM 阅读笔记

- 原文：Huifeng Guo 等，*DeepFM: A Factorization-Machine based Neural Network for CTR Prediction*，2017。[arXiv:1703.04247](https://arxiv.org/abs/1703.04247)；本地 [PDF](Guo_2017_DeepFM.pdf)，8 页，v1。
- 核心问题：CTR 的特征交互既有低阶关系，也可能有复杂高阶关系。只用 FM 或只用深层网络，各有表达侧重。
- 结构：FM 分支建模一阶和二阶交互，深层分支学习非线性组合；两支共享原始特征的 embedding 输入，最终合并预测。
- 建议先读模型结构图与共享 embedding 的说明，再读实验中的对照/消融。重点辨认 FM 分支、MLP 分支及最终输出分别做什么。
- 与仓库关系：目前只有 [FM 原理实验](../../models/fm/README.md)，还没有稀疏字段、MLP 或 DeepFM。下一阶段应在同一数据协议下比较 LR、FM、MLP、DeepFM；提升不能直接归功于任一分支。
- 自检：共享 embedding 与分别训练两套 embedding 有何不同？为什么必须保留单独的 FM 和 MLP 对照？
