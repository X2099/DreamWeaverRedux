# -*- coding: utf-8 -*-
"""
@File    : prepare.py
@Time    : 2024/4/2 17:28
@Desc    : 预处理数据
"""
import os
import glob
import pickle
import re

import opencc


def traditional_simplified():
    """
    繁体转简体
    """
    traditional_root = 'chinese_ancient_books\\traditional'
    simplified_root = 'chinese_ancient_books\\simplified'
    os.makedirs(simplified_root, exist_ok=True)

    converter = opencc.OpenCC('t2s')

    def traditional2simplified_clean(traditional_f, simplified_d):
        """
        繁体文本转简体文本并清洗
        """
        with open(traditional_f, 'r', encoding='utf-8') as tf:
            file_name = traditional_f.split('\\')[-1]
            file_name = os.path.splitext(file_name)[0]
            file_name = converter.convert(file_name) + '.txt'
            simplified_f = os.path.join(simplified_d, file_name)
            content = converter.convert(tf.read())
            content = re.sub(r"<[^>]*>", "", content)
            sentences = content.split('\n')
            cleaned_sentences = []
            for sentence in sentences:
                cleaned_sentence = re.sub(r'\s+', ' ', sentence)
                cleaned_sentence = cleaned_sentence.strip()
                if len(re.findall(r'[\u4e00-\u9fa5]', cleaned_sentence)) < 1:  # 没有汉字
                    continue
                if cleaned_sentence:
                    cleaned_sentences.append(cleaned_sentence + '\n')
            with open(simplified_f, 'w', encoding='utf-8') as sf:
                sf.writelines(cleaned_sentences)

    for root, dirs, files in os.walk(traditional_root):

        if root == traditional_root:
            for file in files:
                file = os.path.join(root, file)
                traditional2simplified_clean(file, simplified_root)
                print(f"繁体转简体：《{file}》 完成。")

        for t_dir_name in dirs:
            s_dir_name = converter.convert(t_dir_name)
            simplified_dir = os.path.join(simplified_root, s_dir_name)
            os.makedirs(simplified_dir, exist_ok=True)

            traditional_dir = os.path.join(traditional_root, t_dir_name)
            traditional_htmls = glob.glob(f'{traditional_dir}\\*.html')
            traditional_texts = glob.glob(f'{traditional_dir}\\*.txt')
            traditional_mds = glob.glob(f'{traditional_dir}\\*.md')

            for traditional_file in traditional_htmls + traditional_texts + traditional_mds:
                traditional2simplified_clean(traditional_file, simplified_dir)
            print(f"繁体转简体：《{t_dir_name}》 -> 《{s_dir_name}》 完成。")


def merge_texts():
    """
    合并所有的数据
    """
    data_dir = 'chinese_ancient_books\\simplified'
    inputs = "input.txt"
    with open(inputs, 'w', encoding='utf-8') as input_f:
        for root, dirs, files in os.walk(data_dir):
            for file in files:
                file = os.path.join(root, file)
                with open(file, 'r', encoding='utf-8') as f:
                    input_f.writelines(f.readlines())


def get_word_list(meta_file='meta.pkl', input_file='input.txt'):
    """
    对全文分词获得词表
    :return:
    """

    with open(input_file, 'r', encoding='utf-8') as input_f:
        words = input_f.read()
        print(f"训练数据总共有 {round(len(words) / 10000, 2)} 万字。")
    words = list(set(words))
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


if __name__ == '__main__':
    # merge_texts()
    get_word_list()
