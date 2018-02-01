#运行环境 python3

import numpy as np
from sklearn.decomposition import PCA
from sklearn.svm import SVC

#设置 PCA 降纬后的纬度值
COMPONENT_NUM = 50

print('Read training data...')

#读取训练数据
with open('train.csv','r') as read:
    #读取文件第一行的表头，不做处理
    read.readline()
    #训练文件的标签集
    train_label = []
    #训练文件的数据集
    train_data = []
    for line in read.readlines():
        #map()是 Python 内置的高阶函数，它接收一个函数 f 和一个 list，
        #并通过把函数 f 依次作用在 list 的每个元素上，得到一个新的 list 并返回
        #把 list 里的所有元素转换为 整型 返回
        data = list( map(int,line.rstrip().split(',')))
        #把 data 里的第一个元素加入到标签集
        train_label.append(data[0])
        #把 data 里剩余的元素加入到数据集
        train_data.append(data[1:])

print('Loaded ' + str(len(train_label)))

print('Reduction...')

# 将 list 转换为 numpy 数组
train_label = np.array(train_label)
train_data = np.array(train_data)

# 原始数据集的维度
print (train_data.shape)
pca = PCA(n_components=COMPONENT_NUM, whiten=True)
# Fit the model with X
pca.fit(train_data)
#对数据集 train_data 进行降纬
train_data = pca.transform(train_data)
#降纬后的数据集纬度
print (train_data.shape )

print('Train SVM...')

svc = SVC()
#训练 SVM
svc.fit(train_data, train_label)

print('Read testing data...')

#加载测试集
with open('test.csv','r') as read:
    #读取文件第一行的表头，不做处理
    read.readline()
    #测试数据集
    test_data = []
    for line in read.readlines():
        #map 把 list 里的所有元素转换为 整型 返回
        data = list(map(int, line.rstrip().split(',')))
        #把所有数据加入测试数据集
        test_data.append(data)

print('Loaded ' + str(len(test_data)))

print('Predicting...')
#将 list 转换为 numpy 数组
test_data = np.array(test_data)
#降纬
test_data = pca.transform(test_data)
#测试结果
predict = svc.predict(test_data)

print('Saving...')

#将测试结果保存到 sample_submission.csv
with open('sample_submission.csv', 'w') as writer:
    #第一行写入表头
    writer.write('"ImageId","Label"\n')
    count = 0
    for p in predict:
        count += 1
        writer.write(str(count) + ',"' + str(p) + '"\n')
