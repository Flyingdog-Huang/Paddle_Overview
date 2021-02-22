import numpy as np
# 加载相关库
import os
import random
import paddle
import numpy as np
from PIL import Image
import gzip
import json
import paddle.nn.functional as F
from paddle.nn import Conv2D, MaxPool2D, Linear

# datafile = 'housing.data'
# data = np.fromfile(datafile, sep=' ')
# # print(np.size(data))

# # 每条数据包括14项，其中前面13项是影响因素，第14项是相应的房屋价格中位数
# feature_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', \
#                       'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV' ]
# feature_num = len(feature_names)

# # 将原始数据进行Reshape，变成[N, 14]这样的形状
# data = data.reshape([data.shape[0]//feature_num, feature_num])
# # print(np.size(data))

# # resize是改变原数组的,但是reshape却不改变.

# ratio = 0.8
# offset = int(data.shape[0] * ratio)
# training_data = data[:offset]

# maximums, minimums, avgs = training_data.max(axis=0), training_data.min(axis=0), \
#                                training_data.sum(axis=0)/training_data.shape[0]

# print(maximums, minimums, avgs)
# # 运算操作时，按数据shape进行

l = [random.random()] * 105
index_list = list(range(len(l)))


def load_data():
    n = 30

    def data_ge():
        l_data = []
        # 按照索引读取数据
        for i in index_list:
            l_data.append(l[i])
            if len(l_data) == n:
                yield np.array(l_data)
                l_data = []
        if len(l_data) > 0:
            yield np.array(l_data)

    return data_ge


g = load_data()
for i in g():
    print(i)

# # 定义数据集读取器
# def load_data(mode='train'):
#     # 读取数据文件
#     datafile = 'mnist.json.gz'
#     print('loading mnist dataset from {} ......'.format(datafile))
#     data = json.load(gzip.open(datafile))
#     # 读取数据集中的训练集，验证集和测试集
#     train_set, val_set, eval_set = data

#     # 数据集相关参数，图片高度IMG_ROWS, 图片宽度IMG_COLS
#     IMG_ROWS = 28
#     IMG_COLS = 28
#     # 根据输入mode参数决定使用训练集，验证集还是测试
#     if mode == 'train':
#         imgs = train_set[0]
#         labels = train_set[1]
#     elif mode == 'valid':
#         imgs = val_set[0]
#         labels = val_set[1]
#     elif mode == 'eval':
#         imgs = eval_set[0]
#         labels = eval_set[1]

#     # 获得所有图像的数量
#     imgs_length = len(imgs)
#     # 验证图像数量和标签数量是否一致
#     assert len(imgs) == len(labels), \
#           "length of train_imgs({}) should be the same as train_labels({})".format(
#                   len(imgs), len(labels))

#     index_list = list(range(imgs_length))

#     # 读入数据时用到的batchsize
#     BATCHSIZE = 100

#     # 定义数据生成器
#     def data_generator():
#         # 训练模式下，打乱训练数据
#         if mode == 'train':
#             random.shuffle(index_list)
#         imgs_list = []
#         labels_list = []
#         # 按照索引读取数据
#         for i in index_list:
#             # 读取图像和标签，转换其尺寸和类型
#             img = np.reshape(imgs[i],
#                              [1, IMG_ROWS, IMG_COLS]).astype('float32')
#             label = np.reshape(labels[i], [1]).astype('int64')
#             imgs_list.append(img)
#             labels_list.append(label)
#             # 如果当前数据缓存达到了batch size，就返回一个批次数据
#             if len(imgs_list) == BATCHSIZE:
#                 yield np.array(imgs_list), np.array(labels_list)
#                 # 清空数据缓存列表
#                 imgs_list = []
#                 labels_list = []

#         # 如果剩余数据的数目小于BATCHSIZE，
#         # 则剩余数据一起构成一个大小为len(imgs_list)的mini-batch
#         if len(imgs_list) > 0:
#             yield np.array(imgs_list), np.array(labels_list)

#     return data_generator

# #调用加载数据的函数
# train_loader = load_data('train')

# images, labels = next(train_loader)
# print(images.shape)
# print(labels.shape)

# for batch_id, data in enumerate(train_loader()):
#     #准备数据
#     images, labels = data
#     print(batch_id)
#     print(len(images))
#     print(len(labels))