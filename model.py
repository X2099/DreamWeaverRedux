# -*- coding: utf-8 -*-
"""
@File    : model.py
@Time    : 2024/4/3 15:17
@Desc    : 
"""
import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class Config:
    batch_size: int = 12  # 批量大小
    block_size: int = 16  # 模型的输入序列长度
    vocab_size: int = 5908  # 模型的词汇量大小
    n_layer: int = 12  # 模型的Transformer堆叠层数
    n_head: int = 8  # 模型的注意力头数
    n_embd: int = 512  # 模型的嵌入维度
    dropout: float = 0.0  # 模型中的dropout概率，丢弃法，防止过拟合
    bias: bool = False  # 是否设置偏置


class LayerNorm(nn.Module):
    """层归一化"""

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, x):
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, 1e-5)


class MultiHeadAttention(nn.Module):
    """多头注意力层"""

    def __init__(self, config: Config):
        super().__init__()

        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head  # 注意力头数
        self.n_embd = config.n_embd  # 词嵌入维度
        self.dropout = config.dropout
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                             .view(1, 1, config.block_size, config.block_size))  # 三角掩码矩阵

    def forward(self, x):
        B, T, C = x.size()  # 批量大小 序列长度 词嵌入维度
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # 进行维度变化，分头
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y


class MLP(nn.Module):
    """多层感知机"""

    def __init__(self, config: Config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)  # 隐藏层
        x = self.gelu(x)  # 激活函数
        x = self.c_proj(x)  # 输出投影层
        x = self.dropout(x)
        return x


class Block(nn.Module):
    """Transformer Block"""

    def __init__(self, config: Config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)  # 层归一化
        self.attn = MultiHeadAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)  # 前馈神经网络

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class NovelGPT(nn.Module):

    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            wpe=nn.Embedding(config.vocab_size, config.n_embd),
            drop=nn.Dropout(config.dropout),
            h=nn.ModuleList(Block(config) for _ in range(config.n_layer)),
            ln_f=LayerNorm(config.n_embd, bias=config.bias)
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.apply(self._init_weights)
        print(f"本模型参数规模：{round(self.get_num_params() / 10000, 2)} 万个。")

    def _init_weights(self, module):
        """
        参数初始化
        """
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        b, t = idx.size()
        pos = torch.arange(0, t, dtype=torch.long)  # 位置，简单粗暴的绝对位置编码
        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            logits = self.lm_head(x[:, [-1], :])
            loss = None
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, temperature=1.0, top_k=None, end_token=0):
        """
        根据提示词进行预测
        """
        while True:
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond)
            # 这行代码的作用是对模型输出的 logits 进行温度调节。在文本生成中，温度调节是一种常见的技术，用于控制模型生成文本的多样性。
            logits = logits[:, -1, :] / temperature
            # 如果设置了 top_k，则将 logits 中排名靠前的 top_k 个词的概率保留，其余设为负无穷，这样在采样时只有这些词有可能被选中。
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(logits, dim=-1)
            # 从多项分布中进行采样1个样本作为输出
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
            if idx_next.item() == end_token:
                break

        return idx

    def get_num_params(self, non_embedding=True):
        """
        计算参数的数量
        """
        num_params = sum(p.numel() for p in self.parameters())
        # 通常情况下，位置嵌入是通过固定的数学函数（例如正弦和余弦函数）生成的，而不是通过学习得到的。
        # 因此，位置嵌入不需要更新或调整，也不会受到梯度的影响。
        if non_embedding:
            num_params -= self.transformer.wpe.weight.numel()
        return num_params


if __name__ == '__main__':
    NovelGPT(Config())
