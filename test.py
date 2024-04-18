# -*- coding: utf-8 -*-
"""
@File    : test.py
@Time    : 2024/4/17 11:48
@Desc    : 
"""
import math

import torch
from torch import nn

torch.manual_seed(2099)

x1 = torch.arange(6)
x2 = torch.arange(11, 5, -1)

x = torch.stack((x1, x2))

print('x:', x.shape)

e = nn.Embedding(12, 8)

print('embedding weight:', e.weight.shape)

x = e(x)

print('embed x:', x.shape)

c_attn = nn.Linear(8, 3 * 8, bias=False)

x = c_attn(x)

print(x.shape)

q, k, v = x.split(8, dim=-1)

print(q.shape, k.shape, v.shape)
# print(k)

q = q.view(2, 6, 2, 8 // 2).transpose(1, 2)
k = k.view(2, 6, 2, 8 // 2).transpose(1, 2)
v = v.view(2, 6, 2, 8 // 2).transpose(1, 2)

print(k.shape)
# print(k)

att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))

print(att.shape)
