import torch


class CustomAdam:
    # 参数介绍 params:模型参数 lr:学习率 betas:一阶矩和二阶矩的指数衰减率 eps:数值稳定性常数 weight_decay:权重衰减系数
    def __init__(
        self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2
    ):
        self.params = list(params)
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0

        # 初始化一阶矩和二阶矩
        self.state = []
        for p in self.params:
            self.state.append(
                {
                    "m": torch.zeros_like(p.data),  # 一阶矩
                    "v": torch.zeros_like(p.data),  # 二阶矩
                }
            )

    # 梯度清零
    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.grad.zero_()

    # 更新参数
    def step(self):
        self.t += 1
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            grad = p.grad.data
            state = self.state[i]

            # 1 L2正则化
            if self.weight_decay != 0:
                p.data.mul_(1 - self.lr * self.weight_decay)#对参数进行权重衰减，乘以(1 - lr * weight_decay)来缩小参数的值，从而实现L2正则化

            # 2 更新有偏炬估计
            state["m"] = self.beta1 * state["m"] + (1 - self.beta1) * grad#更新一阶矩，使用指数衰减平均来计算梯度的平均值
            state["v"] = self.beta2 * state["v"] + (1 - self.beta2) * (grad**2)#更新二阶矩，使用指数衰减平均来计算梯度的平方的平均值
            
            # 3偏差修正
            m_hat = state["m"] / (1 - self.beta1**self.t)#计算一阶矩的偏差修正，除以(1 - beta1^t)来纠正初始阶段的偏差
            v_hat = state["v"] / (1 - self.beta2**self.t)#计算二阶矩的偏差修正，除以(1 - beta2^t)来纠正初始阶段的偏差
            
            # 4 更新参数
            p.data.addcdiv_(m_hat,torch.sqrt(v_hat)+self.eps,value=-self.lr)#使用修正后的矩估计来更新参数，除以sqrt(v_hat)加上eps来避免除零错误，并乘以学习率来控制更新的步长
