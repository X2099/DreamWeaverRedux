# -*- coding: utf-8 -*-
"""
@File    : prepare.py
@Time    : 2024/4/2 17:28
@Desc    : 预处理数据
"""
import os
import glob
import pickle
import opencc


def traditional_simplified(traditional_dir, simplified_dir, split=False):
    """
    繁体转简体
    :return:
    """
    os.makedirs(simplified_dir, exist_ok=True)

    converter = opencc.OpenCC('t2s')
    traditional_htmls = glob.glob(f'{traditional_dir}\\*.html')

    for traditional_html in traditional_htmls:
        with open(traditional_html, 'r', encoding='utf-8') as tf:
            _, file_name = tf.name.split(traditional_dir + '\\')
            file_name = os.path.splitext(file_name)[0]
            file_name = converter.convert(file_name) + '.txt'
            simplified_html = os.path.join(simplified_dir, file_name)
            content = converter.convert(tf.read())
            content = content.replace('<h2>', '').replace('</h2>', '').replace('<p>', '').replace('</p>', '')
            with open(simplified_html, 'w', encoding='utf-8') as sf:
                sf.write(content)
            if split and file_name.split()[0].isdigit() and int(file_name.split()[0]) >= 80:
                break


def split_sentences(data_dir, inputs):
    """
    分割句子
    :return:
    """
    texts = glob.glob(f'{data_dir}\\*.txt')
    input_sentences = []
    for txt_file in texts:
        with open(txt_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            input_sentences.extend(lines)

    with open(inputs, 'w', encoding='utf-8') as f:
        f.writelines(input_sentences)


def merge_novels(novel_files, input_file='input.txt'):
    """
    合并训练数据
    """
    with open(input_file, 'w', encoding='utf-8') as input_f:
        for novel_file in novel_files:
            with open(novel_file, 'r', encoding='utf-8') as novel_f:
                input_f.write(novel_f.read())


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
    pass
    # traditional_simplified("novels/res/紅樓夢", "novels/res/红楼梦", split=True)
    # split_sentences("novels/res/红楼梦", 'novels/红楼梦.txt')

    # traditional_simplified("novels/res/水滸傳", "novels/res/水浒传")
    # split_sentences("novels/res/水浒传", 'novels/水浒传.txt')

    # traditional_simplified("novels/res/西遊記", "novels/res/西游记")
    # split_sentences("novels/res/西游记", 'novels/西游记.txt')

    # traditional_simplified("novels/res/三國演義", "novels/res/三国演义")
    # split_sentences("novels/res/三国演义", 'novels/三国演义.txt')

    # traditional_simplified("novels/res/封神演义", "novels/res/封神演义S")
    # split_sentences("novels/res/封神演义S", 'novels/封神演义.txt')

    # traditional_simplified("novels/res/初刻拍案惊奇", "novels/res/初刻拍案惊奇S")
    # split_sentences("novels/res/初刻拍案惊奇S", 'novels/初刻拍案惊奇.txt')

    # traditional_simplified("novels/res/二刻拍案惊奇", "novels/res/二刻拍案惊奇S")
    # split_sentences("novels/res/二刻拍案惊奇S", 'novels/二刻拍案惊奇.txt')

    # traditional_simplified("novels/res/聊齋志異", "novels/res/聊斋志异")
    # split_sentences("novels/res/聊斋志异", 'novels/聊斋志异.txt')

    # traditional_simplified("novels/res/儒林外史", "novels/res/儒林外史S")
    # split_sentences("novels/res/儒林外史S", 'novels/儒林外史.txt')

    # merge_novels(novel_files=[
    #     'novels/红楼梦.txt',
    #     'novels/水浒传.txt',
    #     'novels/西游记.txt',
    #     'novels/三国演义.txt',
    #     'novels/封神演义.txt',
    #     'novels/初刻拍案惊奇.txt',
    #     'novels/二刻拍案惊奇.txt',
    #     'novels/聊斋志异.txt',
    #     'novels/儒林外史.txt'
    # ])

    get_word_list()
