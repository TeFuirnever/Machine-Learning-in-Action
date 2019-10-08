#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/10/8 14:40
# @Author  : GXl
# @File    : 5.4.1.py
# @Software: win10 Tensorflow1.13.1 python3.5.6


import matplotlib.pyplot as plt
import numpy as np

"""
Parameters:
    无
Returns:
    dataMat - 数据列表
    labelMat - 标签列表
"""
# 函数说明:加载数据
def loadDataSet():
    dataMat = []                                                        #创建数据列表
    labelMat = []                                                       #创建标签列表
    fr = open('testSet.txt')                                            #打开文件
    for line in fr.readlines():                                         #逐行读取
        lineArr = line.strip().split()                                  #去回车，放入列表
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])     #添加数据
        labelMat.append(int(lineArr[2]))                                #添加标签
    fr.close()                                                          #关闭文件
    return dataMat, labelMat                                            #返回

# 函数说明:绘制数据集
def plotDataSet():
    dataMat, labelMat = loadDataSet()                                   #加载数据集
    dataArr = np.array(dataMat)                                         #转换成numpy的array数组
    n = np.shape(dataMat)[0]                                            #数据个数
    xcord1 = []; ycord1 = []                                            #正样本
    xcord2 = []; ycord2 = []                                            #负样本
    for i in range(n):                                                  #根据数据集标签进行分类
        if int(labelMat[i]) == 1:
            xcord1.append(dataArr[i,1]); ycord1.append(dataArr[i,2])    #1为正样本
        else:
            xcord2.append(dataArr[i,1]); ycord2.append(dataArr[i,2])    #0为负样本
    fig = plt.figure()
    ax = fig.add_subplot(111)                                           #添加subplot
    ax.scatter(xcord1, ycord1, s = 20, c = 'red', marker = 's',alpha=.5)#绘制正样本
    ax.scatter(xcord2, ycord2, s = 20, c = 'green',alpha=.5)            #绘制负样本
    plt.title('DataSet')                                                #绘制title
    plt.xlabel('x'); plt.ylabel('y')                                    #绘制label
    plt.show()                                                          #显示


if __name__ == '__main__':
    plotDataSet()