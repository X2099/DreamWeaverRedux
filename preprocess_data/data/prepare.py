# -*- coding: utf-8 -*-
"""
@File    : prepare.py
@Time    : 2024/4/3 11:10
@Desc    : 
"""
import pickle

import jieba

words = set()

with open('input.txt', 'r', encoding='utf-8') as f:
    sentences = f.readlines()
    for sentence in sentences:
        seg_list = jieba.cut(sentence, cut_all=False)
        for word in seg_list:
            words.add(word)

words = list(words)
words.sort()

vocab_size = len(words)

print(f"《红楼梦》中文分词: {vocab_size} 个。")

wtoi = {word: i for i, word in enumerate(words)}
itow = {i: word for i, word in enumerate(words)}


def encode(s):
    return [wtoi[c] for c in s]


def decode(l):
    return ''.join([itow[i] for i in l])


meta = {
    'vocab_size': vocab_size,
    'itow': itow,
    'wtoi': wtoi,
}

with open('meta.pkl', 'wb') as f:
    pickle.dump(meta, f)
