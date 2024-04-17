# -*- coding: utf-8 -*-
"""
@File    : test.py
@Time    : 2024/4/17 11:48
@Desc    : 
"""
import torch
from torch import nn

torch.manual_seed(2099)

e = nn.Embedding(12, 8)

# print(e)
#
print(e.weight.shape)

x1 = torch.arange(6)
x2 = torch.arange(11, 5, -1)

x = torch.stack((x1, x2))

print(x.shape)

x = e(x)

print(x.shape)

c_attn = nn.Linear(8, 3 * 8, bias=False)

x = c_attn(x)

print(x.shape)

q, k, v = x.split(8, dim=-1)

print(q.shape, k.shape, v.shape)
