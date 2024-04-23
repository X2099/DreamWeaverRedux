# -*- coding: utf-8 -*-
"""
@File    : debug.py
@Time    : 2024/4/19 15:45
@Desc    : 
"""
import torch
from torch import nn
from torch.nn import functional as F


# 交叉熵损失是怎么算的
def cross_entropy():
    y = torch.tensor(
        [[12, 9.9, 15],
         [9, 12.5, 8]],
        dtype=torch.float
    )

    targets = torch.tensor([2, 0])
    loss = F.cross_entropy(y, targets, ignore_index=-1)

    print(loss.item())

    print('-' * 100)
    y = F.softmax(y, dim=1)
    print(y)
    y = torch.log(y)
    print(y)
    one_hot_targets = torch.eye(3)[targets]
    print(one_hot_targets)
    res = one_hot_targets * y
    print(res)
    res = -res.sum(dim=1)

    loss_mean = res.sum() / len(res)
    print(loss_mean.item())


def embedding():
    """
    词嵌入
    """
    wte = nn.Embedding(6, 8)
    print(wte.weight)
    print(wte.weight.shape)
    print('-' * 100)

    x = torch.tensor([0, 1, 2, 4, 3, 1])
    x = wte(x)
    print(x)
    print('-' * 100)

    x1 = torch.tensor([0, 1, 2])
    x1 = wte(x1)
    print(x1)
    print('-' * 100)

    x1 = torch.tensor([0, 1, 2, 3, 2, 0, 0])
    x1 = wte(x1)
    print(x1)
    print('-' * 100)


if __name__ == '__main__':
    # cross_entropy()
    embedding()
