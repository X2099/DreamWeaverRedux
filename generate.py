# -*- coding: utf-8 -*-
"""
@File    : generate.py
@Time    : 2024/4/12 21:24
@Desc    : 
"""
import datetime
import torch
from model import Config, AncientChineseGPT
from tools import encode, decode


def main(prompt: str):
    x = encode(prompt)
    x = torch.tensor(x)
    x = x.reshape(-1, x.shape[0])
    model = AncientChineseGPT(Config())
    model.eval()
    today = datetime.datetime.today().strftime('%Y%m%d')
    file_path = f"parameters/parameters-cpu-{today}.pth"
    checkpoint = torch.load(file_path)
    model.load_state_dict(checkpoint)
    end_token = encode('\n')[0]
    y = model.generate(x, end_token=end_token)
    y = [int(i) for i in y[0]]
    return decode(y)


if __name__ == '__main__':
    p = "关羽大战孙悟空"
    generated = main(p)
    print(generated)
