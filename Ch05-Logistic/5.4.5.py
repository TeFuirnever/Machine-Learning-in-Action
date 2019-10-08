#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/10/8 14:43
# @Author  : GXl
# @File    : 5.4.5.py
# @Software: win10 Tensorflow1.13.1 python3.5.6


from matplotlib.font_manager import FontProperties
import matplotlib.pyplot as plt
import numpy as np
import random

'''
Parameters:
    无
Returns:
    dataMat - 数据列表
    labelMat - 标签列表
'''
# 函数说明:加载数据
def loadDataSet():
    dataMat = []                                                    #创建数据列表
    labelMat = []                                                   #创建标签列表
    fr = open('testSet.txt')                                        #打开文件
    for line in fr.readlines():                                     #逐行读取
        lineArr = line.strip().split()                              #去回车，放入列表
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])]) #添加数据
        labelMat.append(int(lineArr[2]))                            #添加标签
    fr.close()                                                      #关闭文件
    return dataMat, labelMat                                        #返回

'''
Parameters:
    inX - 数据
Returns:
    sigmoid函数
'''
# 函数说明:sigmoid函数
def sigmoid(inX):
    return 1.0 / (1 + np.exp(-inX))

'''
Parameters:
    dataMatIn - 数据集
    classLabels - 数据标签
Returns:
'''
# 函数说明:梯度上升算法
def gradAscent(dataMatIn, classLabels):
    dataMatrix = np.mat(dataMatIn)                                  #转换成numpy的mat
    labelMat = np.mat(classLabels).transpose()                      #转换成numpy的mat,并进行转置
    m, n = np.shape(dataMatrix)                                     #返回dataMatrix的大小。m为行数,n为列数。
    alpha = 0.001                                                   #移动步长,也就是学习速率,控制更新的幅度。
    maxCycles = 500                                                 #最大迭代次数
    weights = np.ones((n,1))
    for k in range(maxCycles):
        h = sigmoid(dataMatrix * weights)                           #梯度上升矢量化公式
        error = labelMat - h
        weights = weights + alpha * dataMatrix.transpose() * error
    return weights.getA()                                           #将矩阵转换为数组，返回权重数组

'''
Parameters:
    dataMatrix - 数据数组
    classLabels - 数据标签
    numIter - 迭代次数
Returns:
    weights - 求得的回归系数数组(最优参数)
'''
# 函数说明:改进的随机梯度上升算法
def stocGradAscent1(dataMatrix, classLabels, numIter=150):
    m,n = np.shape(dataMatrix)                                       #返回dataMatrix的大小。m为行数,n为列数。
    weights = np.ones(n)                                             #参数初始化
    for j in range(numIter):
        dataIndex = list(range(m))
        for i in range(m):
            alpha = 4/(1.0+j+i)+0.01                                 #降低alpha的大小，每次减小1/(j+i)。
            randIndex = int(random.uniform(0,len(dataIndex)))        #随机选取样本
            h = sigmoid(sum(dataMatrix[randIndex]*weights))          #选择随机选取的一个样本，计算h
            error = classLabels[randIndex] - h                       #计算误差
            weights = weights + alpha * error * dataMatrix[randIndex]#更新回归系数
            del(dataIndex[randIndex])                                #删除已经使用的样本
    return weights                                                   #返回

'''
Parameters:
    weights_array1 - 回归系数数组1
    weights_array2 - 回归系数数组2
Returns:
    无
'''
# 函数说明:绘制回归系数与迭代次数的关系
def plotWeights(weights_array1,weights_array2):
    #设置汉字格式
    font = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=14)
    #将fig画布分隔成1行1列,不共享x轴和y轴,fig画布的大小为(13,8)
    #当nrow=3,nclos=2时,代表fig画布被分为六个区域,axs[0][0]表示第一行第一列
    fig, axs = plt.subplots(nrows=3, ncols=2,sharex=False, sharey=False, figsize=(20,10))
    x1 = np.arange(0, len(weights_array1), 1)
    #绘制w0与迭代次数的关系
    axs[0][0].plot(x1,weights_array1[:,0])
    axs0_title_text = axs[0][0].set_title(u'梯度上升算法：回归系数与迭代次数关系',FontProperties=font)
    axs0_ylabel_text = axs[0][0].set_ylabel(u'W0',FontProperties=font)
    plt.setp(axs0_title_text, size=20, weight='bold', color='black')
    plt.setp(axs0_ylabel_text, size=20, weight='bold', color='black')
    #绘制w1与迭代次数的关系
    axs[1][0].plot(x1,weights_array1[:,1])
    axs1_ylabel_text = axs[1][0].set_ylabel(u'W1',FontProperties=font)
    plt.setp(axs1_ylabel_text, size=20, weight='bold', color='black')
    #绘制w2与迭代次数的关系
    axs[2][0].plot(x1,weights_array1[:,2])
    axs2_xlabel_text = axs[2][0].set_xlabel(u'迭代次数',FontProperties=font)
    axs2_ylabel_text = axs[2][0].set_ylabel(u'W1',FontProperties=font)
    plt.setp(axs2_xlabel_text, size=20, weight='bold', color='black')
    plt.setp(axs2_ylabel_text, size=20, weight='bold', color='black')


    x2 = np.arange(0, len(weights_array2), 1)
    #绘制w0与迭代次数的关系
    axs[0][1].plot(x2,weights_array2[:,0])
    axs0_title_text = axs[0][1].set_title(u'改进的随机梯度上升算法：回归系数与迭代次数关系',FontProperties=font)
    axs0_ylabel_text = axs[0][1].set_ylabel(u'W0',FontProperties=font)
    plt.setp(axs0_title_text, size=20, weight='bold', color='black')
    plt.setp(axs0_ylabel_text, size=20, weight='bold', color='black')
    #绘制w1与迭代次数的关系
    axs[1][1].plot(x2,weights_array2[:,1])
    axs1_ylabel_text = axs[1][1].set_ylabel(u'W1',FontProperties=font)
    plt.setp(axs1_ylabel_text, size=20, weight='bold', color='black')
    #绘制w2与迭代次数的关系
    axs[2][1].plot(x2,weights_array2[:,2])
    axs2_xlabel_text = axs[2][1].set_xlabel(u'迭代次数',FontProperties=font)
    axs2_ylabel_text = axs[2][1].set_ylabel(u'W1',FontProperties=font)
    plt.setp(axs2_xlabel_text, size=20, weight='bold', color='black')
    plt.setp(axs2_ylabel_text, size=20, weight='bold', color='black')

    plt.show()


if __name__ == '__main__':
    dataMat, labelMat = loadDataSet()
    weights1,weights_array1 = stocGradAscent1(np.array(dataMat), labelMat)

    weights2,weights_array2 = gradAscent(dataMat, labelMat)
    plotWeights(weights_array1, weights_array2)