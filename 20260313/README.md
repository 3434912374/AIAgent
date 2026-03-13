🚀 DeepLearning_Lab: 从优化算法到 Transformer 实践

📌 项目定位本项目是为深度学习进阶设计的教学级框架。通过手写底层 Adam 优化器与手动搭建 Transformer 架构，旨在打破“调包侠”困境，从数学原理和工程架构两个维度深度掌握 AI 开发流程。

🏗️ 项目架构深度解析

1. 核心算法层 (core/optimizers.py)
实现目标：手动实现 Adam (Adaptive Moment Estimation)。
深度学习点：
一阶矩 (Momentum)：利用 $\beta_1$ 平滑梯度，减少震荡。
二阶矩 (RMSProp)：利用 $\beta_2$ 缩放学习率，为稀疏特征提供更大的更新步长。偏差修正 (Bias Correction)：解决训练初期矩估计偏向 0 的问题。

2. 模型架构层 (models/transformer.py)
实现目标：构建用于序列分类的 Transformer Encoder。
深度学习点：
Positional Encoding：利用正余弦函数为无序的 Attention 注入位置信息。
Multi-Head Attention：允许模型在不同的表示子空间里并行学习信息。
残差连接与层归一化 (LayerNorm)：有效缓解深层网络中的梯度消失问题。

3. 工程流水线 (train.py & inference.py)
实现目标：模拟工业级“训练-部署”分离流程。
深度学习点：
掌握 state_dict 的保存与加载，理解模型在 train()（开启 Dropout）与 eval()（关闭 Dropout）模式下的行为差异。

🛠️ 快速上手指南
第一步：环境配置
确保安装了 Python 3.8+ 及以下库：

Bash

pip install torch torchvision matplotlib
第二步：启动训练
运行训练脚本，观察 Loss 下降情况。此时，系统正在调用你编写的 CustomAdam 进行权重更新。

Bash

python train.py
训练完成后，你将在目录下看到 model_weights.pth。

第三步：模型推理
使用训练好的权重对新样本进行预测：

Bash

python inference.py