# -*- coding: utf-8 -*-
"""
@File    : tools.py
@Time    : 2024/4/16 12:02
@Desc    : 
"""
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = ['Microsoft YaHei']
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

with open('data/meta.pkl', 'rb') as f:
    meta = pickle.load(f)


def encode(word_seq):
    w2i = meta['word2index']
    return [w2i[w] for w in word_seq]


def decode(index_seq):
    i2w = meta['index2word']
    return ''.join([i2w[i] for i in index_seq])


def export_data_bin_file():
    """
    把数据保存成二进制文件
    """
    with open('data/input.txt', 'r', encoding='utf-8') as f:
        data = f.read()
        data_ids = encode(data)
        data_ids = np.array(data_ids, dtype=np.uint16)
        data_ids.tofile('train_data.bin')


def plot_training_progress(file='step_loss.pkl'):
    steps = []
    losses = []
    with open(file, 'rb') as slf:
        steps_losses = pickle.load(slf)
    for step, loss in steps_losses:
        steps.append(step)
        losses.append(float(loss))
    plt.figure(figsize=(12.5, 6))
    plt.plot(steps, losses, label='损失')
    plt.xlabel('训练步数')
    plt.ylabel('损失值')
    plt.grid(True)
    plt.subplots_adjust(left=0.05, right=0.95, top=0.9, bottom=0.1)
    plt.savefig('assets/steps_losses.png')


if __name__ == '__main__':
    export_data_bin_file()
    # plot_training_progress()
