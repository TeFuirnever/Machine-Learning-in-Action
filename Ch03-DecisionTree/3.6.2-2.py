#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/10/8 13:59
# @Author  : GXl
# @File    : 3.6.2-2.py
# @Software: win10 Tensorflow1.13.1 python3.5.6


import pandas as pd
from sklearn.preprocessing import LabelEncoder

if __name__ == '__main__':
    with open('lenses.txt', 'r') as fr:                               #加载文件
        lenses = [inst.strip().split('\t') for inst in fr.readlines()]#处理文件
    lenses_target = []                                                #提取每组数据的类别，保存在列表里
    for each in lenses:
        lenses_target.append(each[-1])

    lensesLabels = ['age', 'prescript', 'astigmatic', 'tearRate']     #特征标签
    lenses_list = []                                                  #保存lenses数据的临时列表
    lenses_dict = {}                                                  #保存lenses数据的字典，用于生成pandas
    for each_label in lensesLabels:                                   #提取信息，生成字典
        for each in lenses:
            lenses_list.append(each[lensesLabels.index(each_label)])
        lenses_dict[each_label] = lenses_list
        lenses_list = []
    # print(lenses_dict)                                              #打印字典信息
    lenses_pd = pd.DataFrame(lenses_dict)                             #生成pandas.DataFrame
    print(lenses_pd)                                                  #打印pandas.DataFrame
    le = LabelEncoder()                                               #创建LabelEncoder()对象，用于序列化
    for col in lenses_pd.columns:                                     #为每一列序列化
        lenses_pd[col] = le.fit_transform(lenses_pd[col])
    print(lenses_pd)