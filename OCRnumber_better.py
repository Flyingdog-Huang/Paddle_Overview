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
#引入matplotlib库
import matplotlib.pyplot as plt
from paddle.io import Dataset
#引入VisualDL库，并设定保存作图数据的文件位置
from visualdl import LogWriter


# 创建一个类MnistDataset，继承paddle.io.Dataset 这个类
# MnistDataset的作用和上面load_data()函数的作用相同，均是构建一个迭代器
class MnistDataset(Dataset):
    def __init__(self, mode):
        # 读取数据文件
        datafile = 'mnist.json.gz'
        print('loading mnist dataset from {} ......'.format(datafile))
        data = json.load(gzip.open(datafile))
        # 读取数据集中的训练集，验证集和测试集
        train_set, val_set, eval_set = data

        # 数据集相关参数，图片高度IMG_ROWS, 图片宽度IMG_COLS
        IMG_ROWS = 28
        IMG_COLS = 28
        # 根据输入mode参数决定使用训练集，验证集还是测试
        if mode == 'train':
            # 获得训练数据集
            imgs = train_set[0]
            labels = train_set[1]
        elif mode == 'valid':
            # 获得验证数据集
            imgs = val_set[0]
            labels = val_set[1]
        elif mode == 'eval':
            # 获得测试数据集
            imgs = eval_set[0]
            labels = eval_set[1]
        else:
            raise Exception(
                "mode can only be one of ['train', 'valid', 'eval']")

        # 获得所有图像的数量
        imgs_length = len(imgs)
        # 验证图像数量和标签数量是否一致
        assert len(imgs) == len(labels), \
            "length of train_imgs({}) should be the same as train_labels({})".format(
                    len(imgs), len(labels))

        self.imgs = imgs
        self.labels = labels

    def __getitem__(self, idx):
        img = np.reshape(self.imgs[idx], [1, 28, 28]).astype('float32')
        label = np.reshape(self.labels[idx], [1]).astype('int64')
        return img, label

    def __len__(self):
        return len(self.imgs)


# 定义数据集读取器
def load_data(mode='train'):
    # 读取数据文件
    datafile = 'mnist.json.gz'
    print('loading mnist dataset from {} ......'.format(datafile))
    data = json.load(gzip.open(datafile))
    # 读取数据集中的训练集，验证集和测试集
    train_set, val_set, eval_set = data

    # 数据集相关参数，图片高度IMG_ROWS, 图片宽度IMG_COLS
    IMG_ROWS = 28
    IMG_COLS = 28
    # 根据输入mode参数决定使用训练集，验证集还是测试
    if mode == 'train':
        imgs = train_set[0]
        labels = train_set[1]
    elif mode == 'valid':
        imgs = val_set[0]
        labels = val_set[1]
    elif mode == 'eval':
        imgs = eval_set[0]
        labels = eval_set[1]

    # 获得所有图像的数量
    imgs_length = len(imgs)
    # 验证图像数量和标签数量是否一致
    assert len(imgs) == len(labels), \
          "length of train_imgs({}) should be the same as train_labels({})".format(
                  len(imgs), len(labels))

    index_list = list(range(imgs_length))

    # 读入数据时用到的batchsize
    BATCHSIZE = 100

    # 定义数据生成器
    def data_generator():
        # 训练模式下，打乱训练数据
        if mode == 'train':
            random.shuffle(index_list)
        imgs_list = []
        labels_list = []
        # 按照索引读取数据
        for i in index_list:
            # 读取图像和标签，转换其尺寸和类型
            img = np.reshape(imgs[i],
                             [1, IMG_ROWS, IMG_COLS]).astype('float32')
            label = np.reshape(labels[i], [1]).astype('int64')
            imgs_list.append(img)
            labels_list.append(label)
            # 如果当前数据缓存达到了batch size，就返回一个批次数据
            if len(imgs_list) == BATCHSIZE:
                yield np.array(imgs_list), np.array(labels_list)
                # 清空数据缓存列表
                imgs_list = []
                labels_list = []

        # 如果剩余数据的数目小于BATCHSIZE，
        # 则剩余数据一起构成一个大小为len(imgs_list)的mini-batch
        if len(imgs_list) > 0:
            yield np.array(imgs_list), np.array(labels_list)

    return data_generator


    # 定义模型结构
    # 多层卷积神经网络实现
class MNIST(paddle.nn.Layer):
    def __init__(self):
        super(MNIST, self).__init__()

        # 定义卷积层，输出特征通道out_channels设置为20，卷积核的大小kernel_size为5，卷积步长stride=1，padding=2
        self.conv1 = Conv2D(in_channels=1,
                            out_channels=20,
                            kernel_size=5,
                            stride=1,
                            padding=2)
        # 定义池化层，池化核的大小kernel_size为2，池化步长为2
        self.max_pool1 = MaxPool2D(kernel_size=2, stride=2)
        # 定义卷积层，输出特征通道out_channels设置为20，卷积核的大小kernel_size为5，卷积步长stride=1，padding=2
        self.conv2 = Conv2D(in_channels=20,
                            out_channels=20,
                            kernel_size=5,
                            stride=1,
                            padding=2)
        # 定义池化层，池化核的大小kernel_size为2，池化步长为2
        self.max_pool2 = MaxPool2D(kernel_size=2, stride=2)
        # 定义一层全连接层，输出维度是10
        self.fc = Linear(in_features=980, out_features=10)

    # 定义网络前向计算过程，卷积后紧接着使用池化层，最后使用全连接层计算最终输出
    # 卷积层激活函数使用Relu，全连接层激活函数使用softmax
    def forward(self,
                inputs,
                label=None,
                check_shape=False,
                check_content=False):
        outputs1 = self.conv1(inputs)
        outputs2 = F.sigmoid(outputs1)
        outputs3 = self.max_pool1(outputs2)
        outputs4 = self.conv2(outputs3)
        outputs5 = F.sigmoid(outputs4)
        outputs6 = self.max_pool2(outputs5)
        outputs6 = paddle.reshape(outputs6, [outputs6.shape[0], 980])
        outputs7 = self.fc(outputs6)
        outputs8 = F.softmax(outputs7)

        # 选择是否打印神经网络每层的参数尺寸和输出尺寸，验证网络结构是否设置正确
        if check_shape:
            # 打印每层网络设置的超参数-卷积核尺寸，卷积步长，卷积padding，池化核尺寸
            print(
                "\n########## print network layer's superparams ##############"
            )
            print("conv1-- kernel_size:{}, padding:{}, stride:{}".format(
                self.conv1.weight.shape, self.conv1._padding,
                self.conv1._stride))
            print("conv2-- kernel_size:{}, padding:{}, stride:{}".format(
                self.conv2.weight.shape, self.conv2._padding,
                self.conv2._stride))
            #print("max_pool1-- kernel_size:{}, padding:{}, stride:{}".format(self.max_pool1.pool_size, self.max_pool1.pool_stride, self.max_pool1._stride))
            #print("max_pool2-- kernel_size:{}, padding:{}, stride:{}".format(self.max_pool2.weight.shape, self.max_pool2._padding, self.max_pool2._stride))
            print("fc-- weight_size:{}, bias_size_{}".format(
                self.fc.weight.shape, self.fc.bias.shape))

            # 打印每层的输出尺寸
            print(
                "\n########## print shape of features of every layer ###############"
            )
            print("inputs_shape: {}".format(inputs.shape))
            print("outputs1_shape: {}".format(outputs1.shape))
            print("outputs2_shape: {}".format(outputs2.shape))
            print("outputs3_shape: {}".format(outputs3.shape))
            print("outputs4_shape: {}".format(outputs4.shape))
            print("outputs5_shape: {}".format(outputs5.shape))
            print("outputs6_shape: {}".format(outputs6.shape))
            print("outputs7_shape: {}".format(outputs7.shape))
            print("outputs8_shape: {}".format(outputs8.shape))

        # 选择是否打印训练过程中的参数和输出内容，可用于训练过程中的调试
        if check_content:
            # 打印卷积层的参数-卷积核权重，权重参数较多，此处只打印部分参数
            print(
                "\n########## print convolution layer's kernel ###############"
            )
            print("conv1 params -- kernel weights:", self.conv1.weight[0][0])
            print("conv2 params -- kernel weights:", self.conv2.weight[0][0])

            # 创建随机数，随机打印某一个通道的输出值
            idx1 = np.random.randint(0, outputs1.shape[1])
            idx2 = np.random.randint(0, outputs4.shape[1])

            # 打印卷积-池化后的结果，仅打印batch中第一个图像对应的特征
            print("\nThe {}th channel of conv1 layer: ".format(idx1),
                  outputs1[0][idx1])
            print("The {}th channel of conv2 layer: ".format(idx2),
                  outputs4[0][idx2])
            print("The output of last layer:", outputs8[0], '\n')

        if label is not None:
            acc = paddle.metric.accuracy(input=outputs8, label=label)
            return outputs8, acc
        else:
            return outputs8


#仅优化算法的设置有所差别
def train(model):
    # model = MNIST()
    model.train()
    #调用加载数据的函数
    # train_loader = load_data('train')

    # 声明数据加载函数，使用训练模式，MnistDataset构建的迭代器每次迭代只返回batch=1的数据
    train_dataset = MnistDataset(mode='train')
    # 使用paddle.io.DataLoader 定义DataLoader对象用于加载Python生成器产生的数据，
    # DataLoader 返回的是一个批次数据迭代器，并且是异步的；
    data_loader = paddle.io.DataLoader(train_dataset,
                                       batch_size=100,
                                       shuffle=True)

    #四种优化算法的设置方案，可以逐一尝试效果
    #各种优化算法均可以加入正则化项，避免过拟合，参数regularization_coeff调节正则化项的权重
    # opt = paddle.optimizer.SGD(learning_rate=0.01, parameters=model.parameters())
    # opt = paddle.optimizer.Momentum(learning_rate=0.01, momentum=0.9, parameters=model.parameters())
    # opt = paddle.optimizer.Adagrad(learning_rate=0.01, parameters=model.parameters())
    # opt = paddle.optimizer.Adam(learning_rate=0.01,
    #                             parameters=model.parameters())
    opt = paddle.optimizer.Adam(
        learning_rate=0.001,
        weight_decay=paddle.regularizer.L2Decay(coeff=1e-5),
        parameters=model.parameters())

    iters = []
    losses = []
    iter = 0

    EPOCH_NUM = 5
    for epoch_id in range(EPOCH_NUM):
        # for batch_id, data in enumerate(train_loader()):
        for batch_id, data in enumerate(data_loader()):
            #准备数据
            images, labels = data
            images = paddle.to_tensor(images)
            labels = paddle.to_tensor(labels)

            #前向计算的过程，同时拿到模型输出值和分类准确率
            if batch_id == 0 and epoch_id == 0:
                # 打印模型参数和每层输出的尺寸
                predicts, acc = model(images,
                                      labels,
                                      check_shape=True,
                                      check_content=False)
            elif batch_id == 401:
                # 打印模型参数和每层输出的值
                predicts, acc = model(images,
                                      labels,
                                      check_shape=False,
                                      check_content=True)
            else:
                predicts, acc = model(images, labels)
            #计算损失，取一个批次样本损失的平均值
            loss = F.cross_entropy(predicts, labels)
            avg_loss = paddle.mean(loss)

            #每训练了200批次的数据，打印下当前Loss的情况
            if batch_id % 200 == 0:
                print("epoch: {}, batch: {}, loss is: {}, acc is {}".format(
                    epoch_id, batch_id, avg_loss.numpy(), acc.numpy()))
                # 累计迭代次数和对应的loss
                iters.append(iter)
                losses.append(avg_loss.numpy())
                iter = iter + 200

            #后向传播，更新参数，消除梯度的过程
            avg_loss.backward()
            opt.step()
            opt.clear_grad()

    #保存模型参数
    paddle.save(model.state_dict(), 'mnist_betterTest.pdparams')

    return iters, losses


#创建模型
model = MNIST()
#启动训练过程
iters, losses = train(model)
#画出训练过程中Loss的变化曲线
plt.figure()
plt.title("train loss", fontsize=24)
plt.xlabel("iter", fontsize=14)
plt.ylabel("loss", fontsize=14)
plt.plot(iters, losses, color='red', label='train loss')
plt.grid()
plt.show()


# 加入校验或测试，更好评价模型效果
# 读取上一步训练保存的模型参数，读取校验数据集，并测试模型在校验数据集上的效果
def evaluation(model):
    print('start evaluation .......')
    # 定义预测过程
    params_file_path = 'mnist_betterTest.pdparams'
    # 加载模型参数
    param_dict = paddle.load(params_file_path)
    model.load_dict(param_dict)

    model.eval()
    eval_loader = load_data('eval')

    acc_set = []
    avg_loss_set = []
    for batch_id, data in enumerate(eval_loader()):
        images, labels = data
        images = paddle.to_tensor(images)
        labels = paddle.to_tensor(labels)
        predicts, acc = model(images, labels)
        loss = F.cross_entropy(input=predicts, label=labels)
        avg_loss = paddle.mean(loss)
        acc_set.append(float(acc.numpy()))
        avg_loss_set.append(float(avg_loss.numpy()))

    #计算多个batch的平均损失和准确率
    acc_val_mean = np.array(acc_set).mean()
    avg_loss_val_mean = np.array(avg_loss_set).mean()

    print('loss={}, acc={}'.format(avg_loss_val_mean, acc_val_mean))


# model = MNIST()
# evaluation(model)