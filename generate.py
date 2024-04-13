# -*- coding: utf-8 -*-
"""
@File    : generate.py
@Time    : 2024/4/12 21:24
@Desc    : 
"""
import torch
from model import Config, GPT
from preprocess_data.prepare import encode, decode


def main(prompt: str):
    x = encode(prompt)
    x = torch.tensor(x)
    x = x.reshape(-1, x.shape[0])
    model = GPT(Config())
    model.eval()
    file_path = "parameters.pth"
    checkpoint = torch.load(file_path)
    model.load_state_dict(checkpoint)
    end_token = encode('\n')[0]
    y = model.generate(x, end_token=end_token)
    y = [int(i) for i in y[0]]
    return decode(y)


if __name__ == '__main__':
    p = "黛玉听了，嗤的一笑道：「你既要在这里"
    generated = main(p)
    print(generated)
    """
    黛玉听了，嗤的一笑道：「你既要在这里来了，老娘走。」一诊著，鲍性儿随站道：「宝玉不当辫，你也有歹出物肥鹅饭。务是他两点儿我的和他们说半，可知得得了，老臂叔街，也还有？不是比这赶。耳必是这纸的得起假，只用的一沿最人，你今捶劝不是。我们黛玉在小玩魄子来自悄湘莲妥，不住他同全。只是『凤姐儿无人便摄笑道：「大家仙这鲜众人一河仙，我只是那六般的道，你样儿还不说的说叹茶的，舔们这样？非你不用！」平儿问二，就叫人连站。」又捧了一回浸，
    """
