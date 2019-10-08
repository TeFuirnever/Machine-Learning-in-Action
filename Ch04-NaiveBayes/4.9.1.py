#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/10/8 14:24
# @Author  : GXl
# @File    : 4.9.1.py
# @Software: win10 Tensorflow1.13.1 python3.5.6


# -*- coding: UTF-8 -*-
import os
import jieba

def TextProcessing(folder_path):
    folder_list = os.listdir(folder_path)                  #查看folder_path下的文件
    data_list = []                                         #训练集
    class_list = []

    #遍历每个子文件夹
    for folder in folder_list:
        new_folder_path = os.path.join(folder_path, folder)#根据子文件夹，生成新的路径
        files = os.listdir(new_folder_path)                #存放子文件夹下的txt文件的列表

        j = 1
        #遍历每个txt文件
        for file in files:
            if j > 100:                                    #每类txt样本数最多100个
                break
            with open(os.path.join(new_folder_path, file), 'r', encoding = 'utf-8') as f:#打开txt文件
                raw = f.read()

            word_cut = jieba.cut(raw, cut_all = False)    #精简模式，返回一个可迭代的generator
            word_list = list(word_cut)                    #generator转换为list

            data_list.append(word_list)
            class_list.append(folder)
            j += 1
        print(data_list)
        print(class_list)


if __name__ == '__main__':
    #文本预处理
    folder_path = './SogouC/Sample'                       #训练集存放地址
    TextProcessing(folder_path)