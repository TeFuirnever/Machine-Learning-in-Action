#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/10/8 15:05
# @Author  : GXl
# @File    : 6.5.1.py
# @Software: win10 Tensorflow1.13.1 python3.5.6


# -*-coding:utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np

"""
Parameters:
    fileName - 文件名
Returns:
    dataMat - 数据矩阵
    labelMat - 数据标签
"""
# 读取数据
def loadDataSet(fileName):
    dataMat = []; labelMat = []
    fr = open(fileName)
    for line in fr.readlines():#逐行读取，滤除空格等
        lineArr = line.strip().split('\t')
        dataMat.append([float(lineArr[0]), float(lineArr[1])])#添加数据
        labelMat.append(float(lineArr[2]))#添加标签
    return dataMat,labelMat

"""
数据可视化
Parameters:
    dataMat - 数据矩阵
    labelMat - 数据标签
Returns:
    无
"""
def showDataSet(dataMat, labelMat):
    data_plus = []#正样本
    data_minus = []#负样本
    for i in range(len(dataMat)):
        if labelMat[i] > 0:
            data_plus.append(dataMat[i])
        else:
            data_minus.append(dataMat[i])
    data_plus_np = np.array(data_plus)#转换为numpy矩阵
    data_minus_np = np.array(data_minus)#转换为numpy矩阵
    plt.scatter(np.transpose(data_plus_np)[0], np.transpose(data_plus_np)[1])#正样本散点图
    plt.scatter(np.transpose(data_minus_np)[0], np.transpose(data_minus_np)[1])#负样本散点图
    plt.show()

if __name__ == '__main__':
    dataArr,labelArr = loadDataSet('testSetRBF.txt')#加载训练集
    showDataSet(dataArr, labelArr)