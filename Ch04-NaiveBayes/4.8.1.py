# -*- coding: UTF-8 -*-
import re

# 函数说明:接收一个大字符串并将其解析为字符串列表
def textParse(bigString):                                      #将字符串转换为字符列表
    #将特殊符号作为切分标志进行字符串切分，即非字母、非数字
    listOfTokens = re.split(r'\W*', bigString)
    return [tok.lower() for tok in listOfTokens if len(tok) > 2]#除了单个字母，例如大写的I，其它单词变成小写

'''
Parameters:
    dataSet - 整理的样本数据集
Returns:
    vocabSet - 返回不重复的词条列表，也就是词汇表
'''
# 函数说明:将切分的实验样本词条整理成不重复的词条列表，也就是词汇表
def createVocabList(dataSet):
    vocabSet = set([])                      #创建一个空的不重复列表
    for document in dataSet:
        vocabSet = vocabSet | set(document) #取并集
    return list(vocabSet)

if __name__ == '__main__':
    docList = []; classList = []
    for i in range(1, 26):                                             #遍历25个txt文件
        wordList = textParse(open('email/spam/%d.txt' % i, 'r').read())#读取每个垃圾邮件，并字符串转换成字符串列表
        docList.append(wordList)
        classList.append(1)                                            #标记垃圾邮件，1表示垃圾文件
        wordList = textParse(open('email/ham/%d.txt' % i, 'r').read()) #读取每个非垃圾邮件，并字符串转换成字符串列表
        docList.append(wordList)
        classList.append(0)                                            #标记非垃圾邮件，1表示垃圾文件
    vocabList = createVocabList(docList)                               #创建词汇表，不重复
    print(vocabList)