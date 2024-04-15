# -*- coding: utf-8 -*-
"""
@File    : process_data.py
@Time    : 2024/4/2 17:28
@Desc    : 准备数据
"""
import os
import glob

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


if __name__ == '__main__':
    traditional_simplified("novels/data/紅樓夢", "novels/data/红楼梦", split=True)
    split_sentences("novels/data/红楼梦", 'novels/红楼梦.txt')

    traditional_simplified("novels/data/水滸傳", "novels/data/水浒传")
    split_sentences("novels/data/水浒传", 'novels/水浒传.txt')

    traditional_simplified("novels/data/西遊記", "novels/data/西游记")
    split_sentences("novels/data/西游记", 'novels/西游记.txt')

    traditional_simplified("novels/data/三國演義", "novels/data/三国演义")
    split_sentences("novels/data/三国演义", 'novels/三国演义.txt')
