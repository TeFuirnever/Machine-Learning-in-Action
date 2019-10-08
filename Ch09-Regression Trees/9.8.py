#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/10/8 15:40
# @Author  : GXl
# @File    : 9.8.py
# @Software: win10 Tensorflow1.13.1 python3.5.6


from numpy import *

from tkinter import *
import regTrees

import matplotlib

matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure


# 绘制树
def reDraw(tolS, tolN):
    reDraw.f.clf()  # clear the figure
    reDraw.a = reDraw.f.add_subplot(111)
    if chkBtnVar.get():
        if tolN < 2: tolN = 2
        myTree = regTrees.createTree(reDraw.rawDat, regTrees.modelLeaf, \
                                     regTrees.modelErr, (tolS, tolN))
        yHat = regTrees.createForeCast(myTree, reDraw.testDat, \
                                       regTrees.modelTreeEval)
    else:
        myTree = regTrees.createTree(reDraw.rawDat, ops=(tolS, tolN))
        yHat = regTrees.createForeCast(myTree, reDraw.testDat)
    reDraw.a.scatter(array(reDraw.rawDat[:, 0]), array(reDraw.rawDat[:, 1]), s=5)  # 离散型散点图
    reDraw.a.plot(reDraw.testDat, yHat, linewidth=2.0)  # 构建yHat的连续曲线
    reDraw.canvas.draw()


def getInputs():
    try:
        tolN = int(tolNentry.get())
    except:
        tolN = 10
        print("enter Integer for tolN")
        tolNentry.delete(0, END)
        tolNentry.insert(0, '10')
    try:
        tolS = float(tolSentry.get())
    except:
        tolS = 1.0
        print("enter Float for tolS")
        tolSentry.delete(0, END)
        tolSentry.insert(0, '1.0')
    return tolN, tolS


# 理解用户输入并防止程序崩溃
def drawNewTree():
    tolN, tolS = getInputs()  # 从输入框中获取值
    reDraw(tolS, tolN)  # 生成图


# Tk类型的根部件
root = Tk()

# 创造画布
reDraw.f = Figure(figsize=(5, 4), dpi=100)
# 调用Agg，把Agg呈现在画布上
# Agg是一个C++的库，可以从图像创建光栅图
reDraw.canvas = FigureCanvasTkAgg(reDraw.f, master=root)
reDraw.canvas.draw()
reDraw.canvas.get_tk_widget().grid(row=0, columnspan=3)

Label(root, text="tolN").grid(row=1, column=0)
# 文本输入框1
tolNentry = Entry(root)
tolNentry.grid(row=1, column=1)
tolNentry.insert(0, '10')
Label(root, text="tolS").grid(row=2, column=0)
# 文本输入框2
tolSentry = Entry(root)
tolSentry.grid(row=2, column=1)
tolSentry.insert(0, '1.0')
# 初始化与reDraw()关联的全局变量
Button(root, text="ReDraw", command=drawNewTree).grid(row=1, column=2, rowspan=3)
# 按钮整数值
chkBtnVar = IntVar()
# 复选按钮
chkBtn = Checkbutton(root, text="Model Tree", variable=chkBtnVar)
chkBtn.grid(row=3, column=0, columnspan=2)

reDraw.rawDat = mat(regTrees.loadDataSet('sine.txt'))
reDraw.testDat = arange(min(reDraw.rawDat[:, 0]), max(reDraw.rawDat[:, 0]), 0.01)
reDraw(1.0, 10)

root.mainloop()
