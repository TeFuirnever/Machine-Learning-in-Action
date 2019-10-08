#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/10/8 15:35
# @Author  : GXl
# @File    : 8.6.2-3.py
# @Software: win10 Tensorflow1.13.1 python3.5.6


import numpy as np
from bs4 import BeautifulSoup
import random


# 从页面读取数据，生成retX和retY列表
def scrapePage(retX, retY, inFile, yr, numPce, origPrc):
    """
    Parameters:
        retX - 数据X
        retY - 数据Y
        inFile - HTML文件
        yr - 年份
        numPce - 乐高部件数目
        origPrc - 原价
    Returns:
        无
    """
    # 打开并读取HTML文件
    with open(inFile, encoding='utf-8') as f:
        html = f.read()
    soup = BeautifulSoup(html)
    i = 1
    # 根据HTML页面结构进行解析
    currentRow = soup.find_all('table', r="%d" % i)
    while (len(currentRow) != 0):
        currentRow = soup.find_all('table', r="%d" % i)
        title = currentRow[0].find_all('a')[1].text
        lwrTitle = title.lower()
        # 查找是否有全新标签
        if (lwrTitle.find('new') > -1) or (lwrTitle.find('nisb') > -1):
            newFlag = 1.0
        else:
            newFlag = 0.0
        # 查找是否已经标志出售，我们只收集已出售的数据
        soldUnicde = currentRow[0].find_all('td')[3].find_all('span')
        if len(soldUnicde) == 0:
            print("商品 #%d 没有出售" % i)
        else:
            # 解析页面获取当前价格
            soldPrice = currentRow[0].find_all('td')[4]
            priceStr = soldPrice.text
            priceStr = priceStr.replace('$', '')
            priceStr = priceStr.replace(',', '')
            if len(soldPrice) > 1:
                priceStr = priceStr.replace('Free shipping', '')
            sellingPrice = float(priceStr)
            # 去掉不完整的套装价格
            if sellingPrice > origPrc * 0.5:
                print("%d\t%d\t%d\t%f\t%f" % (yr, numPce, newFlag, origPrc, sellingPrice))
                retX.append([yr, numPce, newFlag, origPrc])
                retY.append(sellingPrice)
        i += 1
        currentRow = soup.find_all('table', r="%d" % i)


# 岭回归
def ridgeRegres(xMat, yMat, lam=0.2):
    """
    Parameters:
        xMat - x数据集
        yMat - y数据集
        lam - 缩减系数
    Returns:
        ws - 回归系数
    """
    xTx = xMat.T * xMat
    denom = xTx + np.eye(np.shape(xMat)[1]) * lam
    if np.linalg.det(denom) == 0.0:
        print("矩阵为奇异矩阵,不能转置")
        return
    ws = denom.I * (xMat.T * yMat)
    return ws


# 依次读取六种乐高套装的数据，并生成数据矩阵
def setDataCollect(retX, retY):
    # 2006年的乐高8288,部件数目800,原价49.99
    scrapePage(retX, retY, './setHtml/lego8288.html', 2006, 800, 49.99)
    # 2002年的乐高10030,部件数目3096,原价269.99
    scrapePage(retX, retY, './setHtml/lego10030.html', 2002, 3096, 269.99)
    # 2007年的乐高10179,部件数目5195,原价499.99
    scrapePage(retX, retY, './setHtml/lego10179.html', 2007, 5195, 499.99)
    # 2007年的乐高10181,部件数目3428,原价199.99
    scrapePage(retX, retY, './setHtml/lego10181.html', 2007, 3428, 199.99)
    # 2008年的乐高10189,部件数目5922,原价299.99
    scrapePage(retX, retY, './setHtml/lego10189.html', 2008, 5922, 299.99)
    # 2009年的乐高10196,部件数目3263,原价249.99
    scrapePage(retX, retY, './setHtml/lego10196.html', 2009, 3263, 249.99)


# 数据标准化
def regularize(xMat, yMat):
    """
    Parameters:
        xMat - x数据集
        yMat - y数据集
    Returns:
        inxMat - 标准化后的x数据集
        inyMat - 标准化后的y数据集
    """
    inxMat = xMat.copy()  # 数据拷贝
    inyMat = yMat.copy()
    yMean = np.mean(yMat, 0)  # 行与行操作，求均值
    inyMat = yMat - yMean  # 数据减去均值
    inMeans = np.mean(inxMat, 0)  # 行与行操作，求均值
    inVar = np.var(inxMat, 0)  # 行与行操作，求方差
    # print(inxMat)
    print(inMeans)
    # print(inVar)
    inxMat = (inxMat - inMeans) / inVar  # 数据减去均值除以方差实现标准化
    return inxMat, inyMat


# 计算平方误差
def rssError(yArr, yHatArr):
    """
    Parameters:
        yArr - 预测值
        yHatArr - 真实值
    Returns:

    """
    return ((yArr - yHatArr) ** 2).sum()


# 计算回归系数w
def standRegres(xArr, yArr):
    """
    Parameters:
        xArr - x数据集
        yArr - y数据集
    Returns:
        ws - 回归系数
    """
    xMat = np.mat(xArr);
    yMat = np.mat(yArr).T
    xTx = xMat.T * xMat  # 根据文中推导的公示计算回归系数
    if np.linalg.det(xTx) == 0.0:
        print("矩阵为奇异矩阵,不能转置")
        return
    ws = xTx.I * (xMat.T * yMat)
    return ws


# 岭回归测试
def ridgeTest(xArr, yArr):
    """
    Parameters:
        xMat - x数据集
        yMat - y数据集
    Returns:
        wMat - 回归系数矩阵
    """
    xMat = np.mat(xArr);
    yMat = np.mat(yArr).T
    # 数据标准化
    yMean = np.mean(yMat, axis=0)  # 行与行操作，求均值
    yMat = yMat - yMean  # 数据减去均值
    xMeans = np.mean(xMat, axis=0)  # 行与行操作，求均值
    xVar = np.var(xMat, axis=0)  # 行与行操作，求方差
    xMat = (xMat - xMeans) / xVar  # 数据减去均值除以方差实现标准化
    numTestPts = 30  # 30个不同的lambda测试
    wMat = np.zeros((numTestPts, np.shape(xMat)[1]))  # 初始回归系数矩阵
    for i in range(numTestPts):  # 改变lambda计算回归系数
        ws = ridgeRegres(xMat, yMat, np.exp(i - 10))  # lambda以e的指数变化，最初是一个非常小的数，
        wMat[i, :] = ws.T  # 计算回归系数矩阵
    return wMat


if __name__ == '__main__':
    lgX = []
    lgY = []
    setDataCollect(lgX, lgY)
    print(ridgeTest(lgX, lgY))
