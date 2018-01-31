#运行环境 python3

import numpy as np
import operator

#数据集
def createDataSet():
    group = np.array([[0,0],[0,0.1],[1.0,1.0],[1.0,1.1]])
    labels = ['B','B','A','A']
    return group,labels

group,labels = createDataSet()

#KNN 算法实现
#用于分类的输入向量 inX,
#输入的训练样本集 dataSet
#标签向量为 labels,
#K 表示用于选择最近邻居的数目
#其中标签向量的元素数目和矩阵 dataSet 的行数相同
def classify0(inX, dataSet, labels, K):

    # 训练数据的个数
    dataSetSize = dataSet.shape[0]

    # 计算欧式距离
    # 将 inX数据，复制成 dataSetSize 个完全一样的数据集，并与 dataSet 数据集进行减运算得到一个新的差集
    diffMat = np.tile(inX, (dataSetSize, 1)) - dataSet

    # 对 diffMat 数据集中的所有元素进行平方运算
    sqDiffMat = diffMat ** 2

    # 对 sqDiffMat 数据中的每行数据都进行求和运算，从而得到一个新的行向量
    sqDistances = sqDiffMat.sum(axis=1)

    #对 sqDistances 中的所有元素执行开方运算
    distances = sqDistances ** 0.5

     #对距离进行排序
     #sortedDistIndicies 为得到的索引的序的列表
    sortedDistIndicies = distances.argsort()

     # 字典集
    classCount = {}

    #确定前 K 个距离最小元素所在的主要分类
    for i in range(K):

        # voteIlabel为得到的label（标签）
        voteIlabel = labels[sortedDistIndicies[i]]

        # 字典classCount中标签voteIlabel对应的值加一， 0代表初始值从0开始
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1

    # classCount.items()： 迭代取出classCount的每个元素
    # key： 需要排序的列表项
    # reverse： 降序排序
                         # python2  items变为iteritems
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)

    #最后返回发生频率最高的元素标签
    return sortedClassCount[0][0]

print (classify0([0.0, 0.0], group, labels, 3))
