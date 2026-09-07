import math

import torch
from torch import nn

from cs336_basics import functions


class Linear(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super(Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.randn(out_features, in_features, device=device, dtype=dtype))
        with torch.no_grad():
            mean = self.weight.mean(dim=-1, keepdim=True)
            std = self.weight.std(dim=-1, keepdim=True)

            self.weight.sub_(mean)
            self.weight.div_(std)
            self.weight.mul_(in_features ** -0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.weight.T


class Embeddings(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super(Embeddings, self).__init__()
        # embedding table.
        self.embed_table = nn.Parameter(torch.randn(num_embeddings, embedding_dim, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.embed_table[x]


class RMSNormal(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super(RMSNormal, self).__init__()
        # learnable gamma parameter.
        self.weight = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        square_sqrt = (x.pow(2).mean(dim=-1,
                                     keepdim=True) + self.eps).sqrt()  # shape = (b,t,emb), but already got the mean value
        square_sqrt = square_sqrt.to(x.device)
        return x * self.weight / square_sqrt


class Swiglu(nn.Module):
    # FFN(x) = SwiGLU(x) = W2(SiLU(W1x) ⊙ W3x)
    def __init__(self, d_model: int, dff=None, device=None, dtype=None):
        super(Swiglu, self).__init__()
        # need make sure the dff is a multiple of 64 to make good use of your hardware.
        if dff is None:
            dff = int(int((8 / 3) * d_model) / 64) * 64
        self.w1 = Linear(d_model, dff, device=device, dtype=dtype)
        self.w2 = Linear(dff, d_model, device=device, dtype=dtype)
        self.w3 = Linear(d_model, dff, device=device, dtype=dtype)
        self.silu = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        silu_parameter = self.w1(x)
        silu_re = torch.sigmoid(silu_parameter) * silu_parameter
        # silu_re = self.w1(silu_re)#torch.sigmoid(x @ self.w1.T)
        o_part = self.w3(x)  # x @ self.w3.T
        silu_mul = silu_re * o_part
        return self.w2(silu_mul)  # sigmoid_mul @ self.w2.T


class RopE(nn.Module):
    def __init__(self, theta: float | int, d_k: int, max_seq_len: int, device=None):
        super(RopE, self).__init__()
        assert d_k % 2 == 0
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len

        # i / theta
        frequencies = theta ** (-torch.arange(0, d_k, 2, dtype=torch.long, device=device) / d_k)
        positions = torch.arange(max_seq_len, device=device)  # 向量
        angles = positions[:, None] * frequencies[
            None, :]  # make angles be a matrix with (seq_len, D/2)， broadcasting rules。
        assert angles.shape == (self.max_seq_len, self.d_k / 2)
        cos = angles.cos()
        sin = angles.sin()
        self.register_buffer('cos', cos, persistent=False)
        self.register_buffer('sin', sin, persistent=False)

    def forward(self, q_or_k: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        cos, sin = self.get_buffer("cos"), self.get_buffer("sin")
        d_k = q_or_k.shape[-1]
        # cos = cos[0:token_positions.shape[-1],:]
        # sin = sin[0:token_positions.shape[-1],:]
        cos = cos[token_positions]  # 并不是每个token_positions 都是从0 开始，需要正确的做 slice.
        sin = sin[token_positions]
        pairs = q_or_k.reshape(*q_or_k.shape[:-1], d_k // 2, 2)
        a = pairs[..., 0]
        b = pairs[..., 1]

        rotated_a = a * cos - b * sin
        rotated_b = a * sin + b * cos
        result = torch.stack([rotated_a, rotated_b], dim=-1).flatten(-2)
        return result


def scale_dot_product_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor,
                                device: str | None = None) -> torch.Tensor:
    d_k = q.shape[-1]
    before_softmax = (q @ k.transpose(-2, -1)) / math.sqrt(d_k)  # shape = (batch,seq_len,seq_len)

    mask = mask[:before_softmax.shape[-2], :before_softmax.shape[-1]]
    mask = mask.to(before_softmax.device)

    before_softmax = torch.masked_fill(before_softmax, ~mask, -torch.inf)
    return functions.softmax(before_softmax, dim=-1) @ v


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, max_seq_len, theta: float | int = 10000, with_rope=False,
                 device=None,
                 dtype=None):
        super(MultiHeadAttention, self).__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model must be divisible by n_heads {n_heads}, got {d_model}")
        self.head_dim = d_model // n_heads

        self.n_heads = n_heads
        self.q_proj = Linear(d_model, d_model, device=device)
        self.k_proj = Linear(d_model, d_model, device=device)
        self.v_proj = Linear(d_model, d_model, device=device)
        self.output_proj = Linear(d_model, d_model,
                                  device=device)  # 我们需要output layer because after concat each heads together，we need shuffle each heads and let them interact
        self.register_buffer("mask", (~(torch.tril(torch.ones(max_seq_len, max_seq_len)) == 0)), persistent=False)

        self.RoPE = RopE(theta=theta, d_k=self.head_dim, max_seq_len=max_seq_len, device=device)
        self.with_rope = with_rope

    def copy_weights_(self, q_proj_weight, k_proj_weight, v_proj_weight, out_proj_weight):
        self.q_proj.weight.copy_(q_proj_weight)
        self.k_proj.weight.copy_(k_proj_weight)
        self.v_proj.weight.copy_(v_proj_weight)
        self.output_proj.weight.copy_(out_proj_weight)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None) -> torch.Tensor:
        # normalization first
        mask = self.get_buffer("mask")
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)
        q_slice = Q.reshape(*Q.shape[:-1], self.n_heads,
                            self.head_dim)  # torch.reshape(Q,:,Q.shape[-1] // self.n_heads, self.n_heads)
        k_slice = K.reshape(*K.shape[:-1], self.n_heads, self.head_dim)
        v_slice = V.reshape(*V.shape[:-1], self.n_heads, self.head_dim)
        attention_dot_results: list[torch.Tensor] = []

        for head_idx in range(self.n_heads):
            q = q_slice[..., head_idx, :]
            k = k_slice[..., head_idx, :]
            v = v_slice[..., head_idx, :]
            if self.with_rope and token_positions is not None:
                q = self.RoPE(q, token_positions)
                k = self.RoPE(k, token_positions)
            attention_dot_results.append(scale_dot_product_attention(q, k, v, mask))

        dot_results_concat = torch.cat(attention_dot_results, dim=-1)
        return self.output_proj(dot_results_concat)


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, theta: float | int, max_seq_len: int, device=None,
                 dtype=None):
        super(TransformerBlock, self).__init__()
        self.ln1 = RMSNormal(d_model, device=device)
        self.ln2 = RMSNormal(d_model, device=device)
        self.attn = MultiHeadAttention(d_model=d_model, n_heads=n_heads, max_seq_len=max_seq_len, theta=theta,
                                       with_rope=True, device=device, dtype=dtype)
        self.ffn = Swiglu(d_model, d_ff, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None) -> torch.Tensor:
        ln1_result = self.ln1(x)
        if token_positions is None:
            token_positions = torch.arange(x.shape[-2], device=x.device)

        attn_re = self.attn(ln1_result, token_positions)

        residual = x + attn_re  # residual_sum
        ln2_result = self.ln2(residual)  # second normalization
        ffn_result = self.ffn(ln2_result)
        return ffn_result + residual


class Embedding(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, device=None, dtype=None):
        super(Embedding, self).__init__()
        self.weight = nn.Parameter(torch.randn(vocab_size, d_model, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weight[x]


class TransformerLM(nn.Module):
    def __init__(self, vocab_size: int, context_length: int, d_model: int, num_layers: int, num_heads: int,
                 d_ff: int, theta: float | int = 10000, device=None, dtype=None):
        super(TransformerLM, self).__init__()
        self.token_embeddings = Embedding(vocab_size,
                                          d_model, device=device, dtype=dtype)
        self.layers = nn.ModuleList([
            TransformerBlock(d_model=d_model, n_heads=num_heads, d_ff=d_ff, theta=theta, max_seq_len=context_length,
                             device=device, dtype=dtype) for _ in range(num_layers)
        ])
        self.context_length = context_length
        self.ln_final = RMSNormal(d_model, device=device)
        self.lm_head = Linear(d_model, vocab_size, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None,
                targets: torch.Tensor | None = None):
        logits = self.token_embeddings(x)  # fancying indexing
        if token_positions is None:
            token_positions = torch.arange(x.shape[-1], device=x.device)

        for layer in self.layers:
            logits = layer(logits, token_positions)

        logits = self.ln_final(logits)
        logits = self.lm_head(logits)
        loss: torch.Tensor
        if targets is None:  #
            return logits
        else:
            loss = functions.cross_entropy_loss(logits, targets)

        return logits, loss
        # return functions.softmax(logits, dim=-1)

    @torch.no_grad()
    def generate(self, input: torch.Tensor, max_new_tokens, endoftext_token: int, temperature: float):
        for _ in range(max_new_tokens):
            idx_cond = input[:, -self.context_length:]
            logits = self(idx_cond)
            logits = logits[:, -1, :]  #
            if temperature > 0:
                logits = logits / temperature
            logits = functions.softmax(logits, dim=-1)  # got probabilities
            next_token = torch.multinomial(logits, 1)
            input = torch.cat((input, next_token), dim=-1)
            if next_token == endoftext_token:
                break
        return input
