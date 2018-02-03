from math import log
import operator

#数据集
def createDataSet():
    dataSet = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    labels = ['no surfacing','flippers']
    return dataSet, labels

#计算给定数据集的香农熵
#熵越高者混合的数据也越多
def calcShannonEnt(dataSet):
    # 数据的中实例的总数
    numEntries = len(dataSet)
    # 创建字典，它的键值是最后一列的数值
    labelCounts = {}
    for f in dataSet:
        # 取出数据集中每组测试数据的最后一个特征值
        currentLabel = f[-1]
        # 如果该特征值不在字典中，者在字典中添加该特征值，初始值为 0
        if currentLabel not in labelCounts.keys(): labelCounts[currentLabel] = 0
        # 对应字典中的特征计数器加一
        labelCounts[currentLabel] += 1
    shannonEnt = 0.0
    # 循环属性值， 计算熵值
    for key in labelCounts:
        # 每个属性的概率值
        prob = float(labelCounts[key])/numEntries
        shannonEnt -= prob * log(prob,2)
    return shannonEnt

d, l = createDataSet()
#得到熵以后，我们就可以按照获取最大信息增益的方法划分数据集
#我们将每个特征划分数据集的结果计算一次信息熵
#然后判断按照哪个特征划分数据集是最好的划分方式


# 按照某个特征划分数据集
# dataSet：待划分的数据集
# axis：划分数据集的特征
# value： 特征的返回值
def splitDataSet(dataSet, axis, value):
    #创建新的列表
    retDataSet = []
    #我们遍历数据集中的每个元素，一旦发现符合要求的值，者将其添加到新创建的列表中
    for featVec in dataSet:
        #将符合特征的数据抽取出来
        if featVec[axis] == value:
            # 拷贝这组数据的值（去掉了该属性的值）
            # featVec[:axis] 只会取出 axis 元素位置之前的所有元素，不会取出 axis
            reducedFeatVec = featVec[:axis]
            # extend 的作用是把  featVec[axis+1:] 里的元素添加仍然以元素的形式添加到到 reducedFeatVec 中
            reducedFeatVec.extend(featVec[axis+1:])
            # append 的作用是把 reducedFeatVec 仍然以列表的形式添加到 retDataSet 中
            retDataSet.append(reducedFeatVec)
    return retDataSet


# 接下来我们将遍历整个数据集，循环计算香农熵和 splitDataSet() 函数，找到最好的特征划分方式
# 熵计算会告诉我们如何划分数据集时最好的数据组织方式
# 在函数长度
#调用函数 splitDataSet、calcShannonEnt时，数据需要满足一定的要求：
#第一：数据必须是一种由列表组成的列表，而且所有的列表元素都有具有相同的数据
#第二：数据的最后一列或者每个实例的最后一个元素是当前实例的类别标签
#无需限定 list 中的数据类型，既可以是数字也可以是字符串，并不影响实际计算


# 选择最好的数据集划分方式
def chooseBestFeatureToSplit(dataSet):
    # 每组元素去掉结果值之后的特征个数
    numFeatures = len(dataSet[0]) - 1
    # 计算数据集原始的熵值
    baseEntropy = calcShannonEnt(dataSet)
    # 初始化信息增益
    bestInfoGain = 0.0
    # 最好的划分特征
    bestFeature = -1
    for i in range(numFeatures):
        # 当前的特征集合
        # example是dataSet的每一组数据， example[i]是每一组数据的第i个元素
        # [example[i] for example in dataSet]就是 由每一组数据的第i个元素组成的集合
        featList = [example[i] for example in dataSet]
        # set去重
        uniqueVals = set(featList)
        newEntropy = 0.0
        # 计算每种划分方式的信息熵
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet)/float(len(dataSet))
            newEntropy += prob * calcShannonEnt(subDataSet)     
        # 计算信息增益
        infoGain = baseEntropy - newEntropy     
        # 找出信息增益最大的值，并记录这个特征的索引值
        if (infoGain > bestInfoGain):       
            bestInfoGain = infoGain        
            bestFeature = i
    return bestFeature

# 返回出现次数最多的分类名称
def majorityCnt(classList):
    classCount={}
    for vote in classList:
        if vote not in classCount.keys(): classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

# 创建树
def createTree(dataSet,labels):
    # dataSet里面每一组数据的最后一个元素的集合
    classList = [example[-1] for example in dataSet]
    # 类别完全相同，则停止划分
    if classList.count(classList[0]) == len(classList): 
        return classList[0]
    # 遍历完所有特征时， 返回出现次数最多的特征
    if len(dataSet[0]) == 1: 
        return majorityCnt(classList)
    # 最好的划分方式
    bestFeat = chooseBestFeatureToSplit(dataSet)
    # 最好的划分标签
    bestFeatLabel = labels[bestFeat]
    # 树的一个节点
    myTree = {bestFeatLabel:{}}
    del(labels[bestFeat])
    # 该特征下的所有属性值
    featValues = [example[bestFeat] for example in dataSet]
    # 去重
    uniqueVals = set(featValues)
    # 遍历属性值
    for value in uniqueVals:
        subLabels = labels[:]
        # 树的分支 = 递归下一层
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value),subLabels)
    return myTree                            

print( createTree(d,l))

# 调用决策树做预测
# inputTree：决策树
# featLabels：测试数据标签
# testVec： 测试数据值
def classify(inputTree,featLabels,testVec):
    #在python2.x中，dict.keys()返回一个列表，
    #在python3.x中，dict.keys()返回一个dict_keys对象
    #比起列表，这个对象的行为更像是set，所以不支持索引的。
    #解决方案：list(dict.keys())[index]
    firstStr = list(inputTree.keys())[0]
    secondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstStr)
    key = testVec[featIndex]
    valueOfFeat = secondDict[key]
    if isinstance(valueOfFeat, dict): 
        classLabel = classify(valueOfFeat, featLabels, testVec)
    else: classLabel = valueOfFeat
    return classLabel

dataSet, featLabels = createDataSet()
inputTree = createTree(dataSet, featLabels)
dataSet, featLabels = createDataSet()
testVec = [1,1]
print (classify(inputTree,featLabels,testVec))

# 因为构建决策树耗时严重， 因此构建成功将决策树保存，然后测试时从文件中直接读取使用
# 将构建的决策树写入文件
def storeTree(inputTree,filename):
    import pickle
    fw = open(filename,'w')
    pickle.dump(inputTree,fw)
    fw.close()

# 从文件中读取决策树
def grabTree(filename):
    import pickle
    fr = open(filename)
    return pickle.load(fr)
