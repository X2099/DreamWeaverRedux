# -*- coding: utf-8 -*-
"""
@File    : attention.py
@Time    : 2024/4/17 11:48
@Desc    : 
"""
import math
from dataclasses import dataclass
from pprint import pprint

import torch
from torch import nn
from torch.nn import functional as F

torch.manual_seed(2099)

words = ['。', '上', '下', '不', '习', '功', '向', '唯', '天', '好', '学', '快', '武', '破', '，']
vocab_size = len(words)

x = ['好好学习，天天向上。', '天下武功，唯快不破。']

print('X:', x)

word2index = {word: i for i, word in enumerate(words)}
index2word = {i: word for i, word in enumerate(words)}

for i, s in enumerate(x):
    x[i] = [word2index[w] for w in s]
x = torch.tensor(x)
print('X:', x.shape)


@dataclass
class Config:
    batch_size: int = 2  # 批量大小
    block_size: int = 10  # 模型的输入序列长度
    vocab_size: int = vocab_size  # 模型的词汇量大小
    n_head: int = 2  # 模型的注意力头数
    n_embd: int = 8  # 模型的嵌入维度
    bias: bool = False  # 是否设置偏置


config = Config()

e = nn.Embedding(config.vocab_size, config.n_embd)

print('embedding weight:', e.weight.shape)

x = e(x)

print('embed X:', x.shape)
B, T, C = x.size()  # 批量大小 序列长度 词嵌入维度

c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=False)

x = c_attn(x)

print('X * 3投影:', x.shape)

q, k, v = x.split(config.n_embd, dim=-1)

print('X在最后一维上分割成 Q，K，V：', q.shape, k.shape, v.shape)

q = q.view(config.batch_size, config.block_size, config.n_head, config.n_embd // config.n_head).transpose(1, 2)
k = k.view(config.batch_size, config.block_size, config.n_head, config.n_embd // config.n_head).transpose(1, 2)
v = v.view(config.batch_size, config.block_size, config.n_head, config.n_embd // config.n_head).transpose(1, 2)

print(f'Q，K，V 分{config.n_head}个头，新增`头数`维度，把`头数`放在`序列长度`前面：', q.shape, k.shape, v.shape)

att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))

print('Q 点乘 K 计算它俩相似度：', att.shape)

bias = torch.tril(torch.ones(config.block_size, config.block_size)).view(1, 1, config.block_size, config.block_size)

print('用于构造的掩码矩阵的三角矩阵：', bias.shape)
print(bias)

att = att.masked_fill(bias[:, :, :T, :T] == 0, float('-inf'))
print('掩码后的相似度矩阵：', att.shape)

att = F.softmax(att, dim=-1)
print('softmax激活后的相似度：', att.shape)

y = att @ v
print(
    '多头注意力权重（相似度） 点乘 V 得到预测值Y\n（每个token得到一个预测值，且每个token的预测值只与这个token所在序列中前面的token相关，'
    '后面的token通过掩码矩阵和softmax使其注意力权重为0）：\n',
    y.shape
)

y = y.transpose(1, 2).contiguous().view(B, T, C)
print('多头注意力合并：', y.shape)

weight = torch.ones(config.n_embd)
bias = torch.zeros(config.n_embd) if config.bias else None
ln = F.layer_norm(y, weight.shape, weight, bias, 1e-5)
print('预测值Y进行层归一化：', y.shape)
lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
logits = lm_head(y)
print(f"预测值Y从{config.n_embd}维 线性投影 到{config.vocab_size}维：", logits.shape)
probs = F.softmax(logits, dim=-1)
print(f"预测值Y使用softmax回归到0~1之间的概率：", probs.shape)
max_prob_id = torch.argmax(probs, dim=-1)
print(f"概率最大的那个下标就是预测值Y：")
print(max_prob_id)
y = []
for row in max_prob_id:
    s = []
    for col in row:
        index = col.item()
        s.append(words[index])
    y.append(''.join(s))
print('Y：', y)

"""
X: ['好好学习，天天向上。', '天下武功，唯快不破。']
X: torch.Size([2, 10])
embedding weight: torch.Size([15, 8])
embed X: torch.Size([2, 10, 8])
X * 3投影: torch.Size([2, 10, 24])
X在最后一维上分割成 Q，K，V： torch.Size([2, 10, 8]) torch.Size([2, 10, 8]) torch.Size([2, 10, 8])
Q，K，V 分2个头，新增`头数`维度，把`头数`放在`序列长度`前面： torch.Size([2, 2, 10, 4]) torch.Size([2, 2, 10, 4]) torch.Size([2, 2, 10, 4])
Q 点乘 K 计算它俩相似度： torch.Size([2, 2, 10, 10])
用于构造的掩码矩阵的三角矩阵： torch.Size([1, 1, 10, 10])
tensor([[[[1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
          [1., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
          [1., 1., 1., 0., 0., 0., 0., 0., 0., 0.],
          [1., 1., 1., 1., 0., 0., 0., 0., 0., 0.],
          [1., 1., 1., 1., 1., 0., 0., 0., 0., 0.],
          [1., 1., 1., 1., 1., 1., 0., 0., 0., 0.],
          [1., 1., 1., 1., 1., 1., 1., 0., 0., 0.],
          [1., 1., 1., 1., 1., 1., 1., 1., 0., 0.],
          [1., 1., 1., 1., 1., 1., 1., 1., 1., 0.],
          [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.]]]])
掩码后的相似度矩阵： torch.Size([2, 2, 10, 10])
softmax激活后的相似度： torch.Size([2, 2, 10, 10])
多头注意力权重（相似度） 点乘 V 得到预测值Y
（每个token得到一个预测值，且每个token的预测值只与这个token所在序列中前面的token相关，后面的token通过掩码矩阵和softmax使其注意力权重为0）：
 torch.Size([2, 2, 10, 4])
多头注意力合并： torch.Size([2, 10, 8])
预测值Y进行层归一化： torch.Size([2, 10, 8])
预测值Y从8维 线性投影 到15维： torch.Size([2, 10, 15])
预测值Y使用softmax回归到0~1之间的概率： torch.Size([2, 10, 15])
概率最大的那个下标就是预测值Y：
tensor([[13, 13, 13,  0, 13,  9,  9,  9,  6,  6],
        [ 9,  6,  6,  6,  6,  6,  6, 13,  6, 13]])
Y： ['破破破。破好好好向向', '好向向向向向向破向破']
"""
