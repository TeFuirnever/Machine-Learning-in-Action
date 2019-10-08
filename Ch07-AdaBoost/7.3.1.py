#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/10/8 15:22
# @Author  : GXl
# @File    : 7.4.1.py
# @Software: win10 Tensorflow1.13.1 python3.5.6


import numpy as np
import matplotlib.pyplot as plt

"""
Parameters:
    无
Returns:
    dataMat - 数据矩阵
    classLabels - 数据标签
"""


# 创建单层决策树的数据集
def loadSimpData():
    datMat = np.matrix([[1., 2.1],
                        [1.5, 1.6],
                        [1.3, 1.],
                        [1., 1.],
                        [2., 1.]])
    classLabels = [1.0, 1.0, -1.0, -1.0, 1.0]
    return datMat, classLabels


"""
Parameters:
    dataMat - 数据矩阵
    labelMat - 数据标签
Returns:
    无
"""


# 数据可视化
def showDataSet(dataMat, labelMat):
    data_plus = []  # 正样本
    data_minus = []  # 负样本
    for i in range(len(dataMat)):
        if labelMat[i] > 0:
            data_plus.append(dataMat[i])
        else:
            data_minus.append(dataMat[i])
    data_plus_np = np.array(data_plus)  # 转换为numpy矩阵
    data_minus_np = np.array(data_minus)  # 转换为numpy矩阵
    plt.scatter(np.transpose(data_plus_np)[0], np.transpose(data_plus_np)[1])  # 正样本散点图
    plt.scatter(np.transpose(data_minus_np)[0], np.transpose(data_minus_np)[1])  # 负样本散点图
    plt.show()


if __name__ == '__main__':
    dataArr, classLabels = loadSimpData()
    showDataSet(dataArr, classLabels)