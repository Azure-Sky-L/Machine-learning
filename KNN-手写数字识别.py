#运行环境 python3

import numpy as np
import operator
from os import listdir

#用于分类的输入向量 inX,
#输入的训练样本集 dataSet
#标签向量为 labels,
#K 表示用于选择最近邻居的数目
#其中标签向量的元素数目和矩阵 dataSet 的行数相同
def classify0(inX, dataSet, labels, K):

    # 训练数据的个数
     # shape[0]代表行数
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


# img数据转向量（二维变一维）
# 把一个32 * 32 的二进制图像矩阵转换为 1 * 1024 的向量
def img2vector(filename):

    #创建 1 * 1024 的 Numpy 数组
    returnVect = np.zeros((1,1024))

    # 打开文件
    fr = open(filename)

    # 循环读取文件的前32行数据
    for i in range(32):

        lineStr = fr.readline()

        # 并将每行的头 32 个字符存储在Numpy数组中
        for j in range(32):
            returnVect[0,32 * i + j] = int(lineStr[j])
    return returnVect


# 手写数字识别测试
def handwritingClassTest():

    # 手写数字的标签集
    hwLabels = []

    # 获取训练目录文件里面的文件列表
    trainingFileList = listdir('trainingDigits')

    # 训练目录里面的文件个数
    m = len(trainingFileList)
    # 创建 m 行 1024 列 的内容为0的矩阵
    trainingMat = np.zeros((m,1024))
    # 循环取出每一个文件
    for i in range(m):
        # 获取每一个文件名
        fileNameStr = trainingFileList[i]
        # 截取从开始位置到“.”的字符串
        fileStr = fileNameStr.split('.')[0] 
        # 截取从开始位置到“_”的字符串， 获得的是文件的标签值
        classNumStr = int(fileStr.split('_')[0])
        # 将标签值存到标签集hwLabels中
        hwLabels.append(classNumStr)
        # 将文件的内容，存到训练集中
        trainingMat[i,:] = img2vector('trainingDigits/%s' % fileNameStr)

    # 测试集
    testFileList = listdir('testDigits')  
    errorCount = 0.0
    # 测试集文件数目
    mTest = len(testFileList)
    for i in range(mTest):
        # 取出文件名
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0] 
        # 测试数据的标签值
        classNumStr = int(fileStr.split('_')[0])
        # 测试数据的图片值
        vectorUnderTest = img2vector('testDigits/%s' % fileNameStr)
        # 预测
        classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3)
        print ("the classifier came back with: %d, the real answer is: %d" % (classifierResult, classNumStr))
        # 累计错误数据
        if (classifierResult != classNumStr): errorCount += 1.0
    print ("\nthe total number of errors is: %d" % errorCount)
    print ("\nthe total error rate is: %f" % (errorCount/float(mTest)))
    print ("\nthe total ture rate is: %f" % (1 - errorCount/float(mTest)))
handwritingClassTest()
