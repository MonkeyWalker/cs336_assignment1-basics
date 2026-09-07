import torch


def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    max = x.max(dim=dim, keepdim=True)[0]
    sub_max = x - max
    return torch.exp(sub_max) / sub_max.exp().sum(dim=dim, keepdim=True)

def softmax_with_temperature(logits: torch.Tensor, temperature: float,dim:int) -> torch.Tensor:
    max = logits.max(dim=dim, keepdim=True)[0]
    sub_max = (logits - max) / temperature
    return torch.exp(sub_max) / sub_max.exp().sum(dim=dim, keepdim=True)
#
# def top_p_sampling(logits: torch.Tensor, target_probability) -> torch.Tensor:
#     torch.nuclear_norm(logits)
#     return
#


# log_softmax = xi - SUM(exp(x))
def log_softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    max = x.max(dim=dim, keepdim=True)[0]
    sum_exp = torch.sum(torch.exp(x - max), dim=dim, keepdim=True)
    return (x - max) - torch.log(sum_exp)


"""
use softmax to calculate the probabilities
use torch.gather to pick up all intent possiblities
then use mean to got scaler result/
"""


def cross_entropy_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    softmax_result = log_softmax(logits, -1)  # $$torch.log(softmax(logits, dim=-1))
    print(softmax_result.device, labels.device, labels.dtype)
    gather_result = torch.gather(softmax_result, dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
    return -gather_result.mean()

def perplexity_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    softmax_result = log_softmax(logits, -1)  # $$torch.log(softmax(logits, dim=-1))
    gather_result = torch.gather(softmax_result, dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
    gather_result_avg = torch.sum(gather_result, dim=-1, keepdim=True) / gather_result.shape[-1]
    return torch.exp(gather_result_avg)