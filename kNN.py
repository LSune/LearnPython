#coding=utf-8
from numpy import *
import operator

def createDataSet():
    group = array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels = ['A','A','B','B']
    return group,labels

def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]# 查看行数
    diffMat = tile(inX, (dataSetSize,1)) - dataSet#tile可以构造出datasetsize行1列的矩阵出来。
    sqDiffMat = diffMat ** 2
    sqDistances = sqDiffMat.sum(axis=1)#将一个矩阵的每一行向量相加
    distances = sqDistances**0.5
    sortedDistIndicies = distances.argsort()
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1#词典默认值为0，每次统计+1
    sortedClassCount = sorted(classCount.iteritems(),
                              key=operator.itemgetter(1),
                              reverse=True)
    '''sorted 语法：
            sorted(iterable[, cmp[, key[, reverse]]])
            参数说明：
            iterable -- 可迭代对象。
            cmp -- 比较的函数，这个具有两个参数，参数的值都是从可迭代对象中取出，此函数必须遵守的规则为，大于则返回1，小于则返回-1，等于则返回0。
            key -- 主要是用来进行比较的元素，只有一个参数，具体的函数的参数就是取自于可迭代对象中，指定可迭代对象中的一个元素来进行排序。
            reverse -- 排序规则，reverse = True 降序 ， reverse = False 升序（默认）。'''

    '''例子

            a = [1,2,3] 
            >>> b=operator.itemgetter(1)      //定义函数b，获取对象的第1个域的值
            >>> b(a) 
            
            2
            
            >>> b=operator.itemgetter(1,0)  //定义函数b，获取对象的第1个域和第0个的值
            >>> b(a) 
            (2, 1)
            
            要注意，operator.itemgetter函数获取的不是值，而是定义了一个函数，通过该函数作用到对象上才能获取值。
            
            sorted函数用来排序，sorted(iterable[, cmp[, key[, reverse]]])
            
            其中key的参数为一个函数或者lambda函数。所以itemgetter可以用来当key的参数
            
            a = [('john', 'A', 15), ('jane', 'B', 12), ('dave', 'B', 10)]
            
            根据第二个域和第三个域进行排序
            
            sorted(students, key=operator.itemgetter(1,2))'''
    return sortedClassCount[0][0]
