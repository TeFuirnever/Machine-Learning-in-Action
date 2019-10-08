#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/10/8 15:37
# @Author  : GXl
# @File    : 9.4.1.py
# @Software: win10 Tensorflow1.13.1 python3.5.6


from numpy import *
import matplotlib.pyplot as plt

# 函数说明:加载数据
"""
Parameters:
	filename - 文件名
"""
def loadDataSet(fileName):
    dataMat = []
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        fltLine = list(map(float, curLine))
        dataMat.append(fltLine)
    return dataMat

# 函数说明:绘制数据集分布
"""
Parameters:
	filename - 文件名
"""
def plotDataSet(filename):
    dataMat = loadDataSet(filename)
    n = len(dataMat)  # 数据个数
    xcord = []
    ycord = []  # 样本点
    for i in range(n):
        xcord.append(dataMat[i][0])
        ycord.append(dataMat[i][1])  # 样本点
    fig = plt.figure()
    ax = fig.add_subplot(111)  # 添加subplot
    ax.scatter(xcord, ycord, s=20, c='blue', alpha=.5)  # 绘制样本点
    plt.title('DataSet')  # 绘制title
    plt.xlabel('X')
    plt.show()


if __name__ == '__main__':
    filename = 'ex00.txt'
    plotDataSet(filename)
