# -*- coding: utf-8 -*-
"""
@File    : train.py
@Time    : 2024/4/10 12:04
@Desc    : 
"""
import os
import datetime

import torch
import numpy as np
from model import Config, NovelGPT

device = 'cuda' if torch.cuda.is_available() else 'cpu'
device_type = 'cuda' if 'cuda' in device else 'cpu'

lr = 0.0001

train_data = np.memmap('train_data.bin', dtype=np.uint16, mode='r')


def get_batch(batch_size, block_size):
    """
    随机获得一个小批量训练样本
    """
    data = train_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i + block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i + 1:i + 1 + block_size]).astype(np.int64)) for i in ix])
    if device_type == 'cuda':
        # 将x, y异步移动到 GPU
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y


def main():
    today = datetime.datetime.today().strftime('%Y%m%d')
    config = Config()
    model = NovelGPT(config)
    file_path = f"parameters/parameters-{device}-{today}.pth"
    if os.path.exists(file_path):
        checkpoint = torch.load(file_path)
        model.load_state_dict(checkpoint)
    model.to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-8)
    for i in range(1, 2001):
        x, y = get_batch(config.batch_size, config.block_size)
        optimizer.zero_grad()
        y_hat, loss = model(x, y)
        loss.backward()
        optimizer.step()
        if i % 100 == 0 or i == 1:
            file_path = f"parameters/parameters-{device}-{today}.pth"
            torch.save(model.state_dict(), file_path)
            print(f"{i} loss = ", loss.item())
            break


if __name__ == '__main__':
    main()
