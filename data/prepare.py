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


def get_word_list(novel_files=None, meta_file='data/meta.pkl', input_file='input.txt', refresh=False):
    """
    对全文分词获得词表
    :return:
    """
    if not refresh and os.path.exists(meta_file):
        with open(meta_file, 'rb') as f:
            meta = pickle.load(f)
    else:
        words = set()
        with open(input_file, 'w', encoding='utf-8') as input_f:
            for novel_file in novel_files:
                with open(novel_file, 'r', encoding='utf-8') as novel_f:
                    novel_words = novel_f.read()
                    input_f.write(novel_words)
                words.update(novel_words)
        vocab_size = len(words)
        print(f"训练数据共有 {vocab_size} 个 token 。")
        word2index = {word: i for i, word in enumerate(words)}
        index2word = {i: word for i, word in enumerate(words)}
        meta = {
            'vocab_size': vocab_size,
            'index2word': index2word,
            'word2index': word2index,
        }
        with open(meta_file, 'wb') as f:
            pickle.dump(meta, f)
    return meta


words_meta = get_word_list(
    novel_files=[
        'novels/三国演义.txt',
        'novels/水浒传.txt',
        'novels/红楼梦.txt',
        'novels/西游记.txt',
    ],
    refresh=False
)


def encode(word_seq):
    w2i = words_meta['word2index']
    return [w2i[w] for w in word_seq]


def decode(index_seq):
    i2w = words_meta['index2word']
    return ''.join([i2w[i] for i in index_seq])


def export_data_bin_file():
    """
    把数据保存成二进制文件
    """
    with open('input.txt', 'r', encoding='utf-8') as f:
        data = f.read()
        data_ids = encode(data)
        data_ids = np.array(data_ids, dtype=np.uint16)
        data_ids.tofile(os.path.join(os.path.dirname(__file__), 'data.bin'))


if __name__ == '__main__':
    export_data_bin_file()
