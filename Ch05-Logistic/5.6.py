#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/10/8 14:49
# @Author  : GXl
# @File    : 5.6.py
# @Software: win10 Tensorflow1.13.1 python3.5.6


from sklearn.linear_model import LogisticRegression

# 函数说明:使用Sklearn构建Logistic回归分类器
def colicSklearn():
    frTrain = open('horseColicTraining.txt')              #打开训练集
    frTest = open('horseColicTest.txt')                   #打开测试集
    trainingSet = []; trainingLabels = []
    testSet = []; testLabels = []
    for line in frTrain.readlines():
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(len(currLine)-1):
            lineArr.append(float(currLine[i]))
        trainingSet.append(lineArr)
        trainingLabels.append(float(currLine[-1]))
    for line in frTest.readlines():
        currLine = line.strip().split('\t')
        lineArr =[]
        for i in range(len(currLine)-1):
            lineArr.append(float(currLine[i]))
        testSet.append(lineArr)
        testLabels.append(float(currLine[-1]))
    classifier = LogisticRegression(solver='liblinear',max_iter=10).fit(trainingSet, trainingLabels)
    test_accurcy = classifier.score(testSet, testLabels) * 100
    print('正确率:%f%%' % test_accurcy)


if __name__ == '__main__':
    colicSklearn()