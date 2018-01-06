from numpy import *
import operator
def createDataSet():
    group = array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels = ['A','A','B','B']
    return group,labels
def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    diffMat = tile(inX, (dataSetSize,1)) - dataSet
    sqDiffMat = diffMat ** 2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances**0.5
    sortedDistIndicies = distances.argsort()
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
	classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1
    sortedClassCount = sorted(classCount.iteritems(),
    key = operator.itemgetter(1), reverse = True)
    return sortedClassCount[0][0]
def file(fi):
    fr = open(fi)
    ar = fr.readlines()
    num = len(ar)
    re = zeros((num,3))
    cl = []
    ind = 0
    for i in ar:
	i = i.strip()
	li = i.split('\t')
 	re[ind,:]  = li[0:3]
	cl.append(int(li[-1]))
	lc += 1
    return re,cl
def au(data):
    minv = data.min(0)
    maxv = data.max(0)
    ran = maxv - minv
    nor = zeros(shape(data))
    m = data.shape[0]
    nor = data - tile(minv, (m,1))
    nor = nor / tile(ranges, (m,1))
    return nor,ranges,minv
