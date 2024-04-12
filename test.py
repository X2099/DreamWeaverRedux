# -*- coding: utf-8 -*-
"""
@File    : train.py
@Time    : 2024/4/10 12:04
@Desc    : 
"""
from model import Config, GPT
from preprocess_data.prepare import load_data

config = Config()
model = GPT(config)

for x, y in load_data(config.batch_size, config.block_size):
    y_hat, loss = model(x, y)
    print("loss = ", loss.item())
    break
