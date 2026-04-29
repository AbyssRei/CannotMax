import torch
import torch.optim as optim


class Muon(torch.optim.Optimizer):
    """
    根据《Muon is Scalable for LLM Training》技术报告实现的 Muon 优化器。
    该优化器使用 Newton-Schulz 迭代来进行矩阵正交化，并自带按矩阵形状调整更新率及权重衰减功能。
    仅适用于维度 >= 2 的张量（即矩阵或张量）。
    """

    def __init__(self, params, lr=1e-3, momentum=0.95, weight_decay=0.1, ns_steps=5):
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay, ns_steps=ns_steps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            weight_decay = group['weight_decay']
            ns_steps = group['ns_steps']

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError('Muon 不支持稀疏梯度计算')

                state = self.state[p]
                if len(state) == 0:
                    state['momentum_buffer'] = torch.zeros_like(grad)

                buf = state['momentum_buffer']

                # 应用基于 Nesterov 的动量: M_t = \mu * M_{t-1} + \nabla L_t
                buf.mul_(momentum).add_(grad)

                # Nesterov 动量前馈给 Newton-Schulz 的输入
                g = grad + momentum * buf

                original_shape = g.shape
                # 将超过两维的参数打平为二维矩阵 (例如在卷积时)
                if g.ndim > 2:
                    g = g.view(g.size(0), -1)

                # Newton-Schulz 迭代参数
                a, b, c = 3.4445, -4.7750, 2.0315

                X = g / (g.norm() + 1e-8)

                # 根据矩阵的宽和高选择乘法顺序，避免巨大的中间矩阵引发 OOM
                if X.size(0) > X.size(1):
                    for _ in range(ns_steps):
                        A = X.T @ X
                        B = X @ A
                        C = B @ A
                        X = a * X + b * B + c * C
                else:
                    for _ in range(ns_steps):
                        A = X @ X.T
                        B = A @ X
                        C = A @ B
                        X = a * X + b * B + c * C

                # 将得到的正交化矩阵恢复原维度
                X = X.view(original_shape)

                # 动态计算更新缩放倍率 (Adjusted LR机制)，匹配 AdamW RMS
                dim0, dim1 = X.size(0), X.size(1) if X.ndim > 1 else 1
                max_dim = max(dim0, dim1)
                scale = 0.2 * (max_dim ** 0.5)

                # W_t = W_{t-1} - \eta_t * (0.2 * O_t * \sqrt{\max(A, B)} + \lambda * W_{t-1})
                # 先执行权重衰减
                p.data.mul_(1 - lr * weight_decay)
                # 应用正交化更新步
                p.data.add_(X, alpha=-lr * scale)

        return loss


def get_muon_adamw_optimizers(model, lr, weight_decay=0.1, muon_momentum=0.95):
    """
    提供标准的组合优化器分发策略。
    对模型的所有全连接（>=2D）参数采用 Muon；
    对包含 embedding 字段或一维参数（如 Bias, LayerNorm）使用 AdamW。
    """
    muon_params = []
    adamw_params = []

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        # Moonlight 技术报告中推荐针对全连接矩阵等二维及以上参数使用 Muon
        # Embedding 参数通常会因大量行未被激活而更新稀疏，更适合用 AdamW
        if p.ndim >= 2 and 'embed' not in name.lower():
            muon_params.append(p)
        else:
            adamw_params.append(p)

    muon_opt = Muon(muon_params, lr=lr, momentum=muon_momentum, weight_decay=weight_decay)
    adamw_opt = optim.AdamW(adamw_params, lr=lr, weight_decay=weight_decay)

    return muon_opt, adamw_opt