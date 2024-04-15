# -*- coding: utf-8 -*-
"""
@File    : train.py
@Time    : 2024/4/10 12:04
@Desc    : 
"""
import datetime

import torch
import numpy as np
from model import Config, NovelGPT

lr = 0.0001

train_data = np.memmap('data/data.bin', dtype=np.uint16, mode='r')


def get_batch(batch_size, block_size):
    """
    随机获得一个小批量训练样本
    """
    data = train_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i + block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i + 1:i + 1 + block_size]).astype(np.int64)) for i in ix])
    return x, y


def main():
    today = datetime.datetime.today().strftime('%Y%m%d')
    config = Config()
    model = NovelGPT(config)
    file_path = f"parameters/parameters-{today}.pth"
    checkpoint = torch.load(file_path)
    model.load_state_dict(checkpoint)
    model.train()
    # optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-8)
    for i in range(1, 101):
        x, y = get_batch(config.batch_size, config.block_size)
        optimizer.zero_grad()
        y_hat, loss = model(x, y)
        loss.backward()
        optimizer.step()
        if i % 100 == 0:
            file_path = f"parameters/parameters-{today}.pth"
            torch.save(model.state_dict(), file_path)
        print(f"{i} loss = ", loss.item())


if __name__ == '__main__':
    main()
