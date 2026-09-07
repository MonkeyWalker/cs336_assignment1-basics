import math
from typing import Iterable, Callable

import numpy
import torch
from mpmath.functions.zetazeros import gram_index


class SGD(torch.optim.Optimizer):
    def __init__(self, params: Iterable[torch.Tensor], lr: float = 1e-4):
        super().__init__(params, {
            "lr": lr
        })

    def step(self, closure: Callable[[], float] | None = None) -> float | None:
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                t = state.get("step", 0)
                eff_learning_rate = lr / numpy.sqrt(t + 1)
                p.data.sub_(eff_learning_rate * p.grad.data)
                state["step"] = t + 1
        return loss


"""
# AdamW core formulas # 
# g = grad 
# 
# 一阶方向
# m = beta1 * m + (1 - beta1) * g 
# 二阶速度 （最近的波动有多大）
# v = beta2 * v + (1 - beta2) * g^2 
# 
# bias correction: `修正训练初期一阶矩和二阶矩估计被初始化为 0 所造成的系统性低估`
# m_hat = m / (1 - beta1^t) 
# v_hat = v / (1 - beta2^t) 
# 
# theta *= (1 - lr * weight_decay) 
# decoupled weight decay 
# theta -= lr * m_hat / (sqrt(v_hat) + eps)  
# 
# Important:
# AdamW does NOT do: 
# g += weight_decay * theta
"""


class AdamW(torch.optim.Optimizer):
    def __init__(self, params: Iterable[torch.Tensor], lr: float = 1e-4, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0.01,device: str | None = None) -> None:
        super().__init__(params, {
            "lr": lr,
            "betas": (betas[0], betas[1]),
            "eps": eps,
            "weight_decay": weight_decay
        })

    # def step(self, closure: Callable[[], float] | None = None) -> float | None:
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr = group["lr"]
            betas = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                t = state.get("step", 1)  # Don't init the step as 0 by default.
                m = state.get("m", 0)
                v = state.get("v", 0)
                beta1, beta2 = betas[0], betas[1]
                m = beta1 * m + (1 - beta1) * p.grad.data
                v = beta2 * v + (1 - beta2) * p.grad.data ** 2
                m_hat = m / (1 - beta1 ** t)
                v_hat = v / (1 - beta2 ** t)
                theta = p.data * (1 - lr * weight_decay)
                theta -= lr * m_hat / (torch.sqrt(v_hat) + eps)
                p.data = theta
                state["step"] = t + 1
                state["m"] = m
                state["v"] = v

        return loss


def get_lr_cosine_schedule(t: int, amax: int | float, amin: int | float, tw: int, tc: int) -> float:
    if t < tw:
        return (t / tw) * amax
    elif tw <= t <= tc:
        cos_part = ((t - tw) * math.pi) / (tc - tw)
        return amin + 0.5 * (1 + numpy.cos(cos_part)) * (amax - amin)
    else:
        return amin





def gradient_clipping_v2(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float, eps: float = 1e-6):
    #gradients: list[torch.Tensor] = []
    sum_squares = 0
    for p in parameters:
        if p.grad is None:
            continue
        square_sum = torch.sum(torch.pow(p.grad.data, 2))
        sum_squares += square_sum
    l2_norm = numpy.sqrt(sum_squares)
    if l2_norm  < max_l2_norm:
        return
    scale = max_l2_norm / (l2_norm + eps)
    for p in parameters:
        if p.grad is None:
            continue
        p.grad.data = p.grad.data * scale
