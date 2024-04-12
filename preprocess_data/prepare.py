# -*- coding: utf-8 -*-
"""
@File    : prepare.py
@Time    : 2024/4/3 11:10
@Desc    : 
"""
import os
import pickle

import numpy as np
import torch
import jieba


def get_word_list(meta_file=r'D:\Coding\mycode\DreamWeaverRedux\preprocess_data\meta.pkl'):
    """
    对全文分词获得词表
    :return:
    """
    if os.path.exists(meta_file):
        with open(meta_file, 'rb') as f:
            meta = pickle.load(f)
    else:
        with open(r'D:\Coding\mycode\DreamWeaverRedux\preprocess_data\input.txt', 'r', encoding='utf-8') as f:
            words = jieba.lcut(f.read(), cut_all=False)
        words = list(set(words))
        words.sort()
        word2index = {word: i for i, word in enumerate(words)}
        index2word = {i: word for i, word in enumerate(words)}
        meta = {
            'vocab_size': len(words),
            'index2word': index2word,
            'word2index': word2index,
        }
        with open(meta_file, 'wb') as f:
            pickle.dump(meta, f)
    return meta


words_meta = get_word_list()


def encode(word_seq):
    w2i = words_meta['word2index']
    return [w2i[w] for w in word_seq]


def decode(index_seq):
    i2w = words_meta['index2word']
    return ''.join([i2w[i] for i in index_seq])


def split_sequence_into_batches(sequence, batch_size, sequence_length):
    num_batches = len(sequence) // (batch_size * sequence_length)
    sequence = sequence[:num_batches * batch_size * sequence_length]  # 截取整数个 batch 的序列

    # 将序列重塑为 Batch Size x Sequence Length 的张量
    batches = np.reshape(sequence, (batch_size, -1, sequence_length))

    return batches


def load_data(batch_size, block_size):
    with open(r'D:\Coding\mycode\DreamWeaverRedux\preprocess_data\input.txt', 'r', encoding='utf-8') as f:
        words = jieba.lcut(f.read(), cut_all=False)
        words = encode(words)
        words = torch.tensor(words, dtype=torch.long)
        for i in range(len(words) - block_size):
            x = torch.stack([words[i:i + block_size] for _ in range(batch_size)])
            y = torch.stack([words[i + 1:i + 1 + block_size] for _ in range(batch_size)])
            yield x, y


if __name__ == '__main__':
    load_data(4, 1024)
