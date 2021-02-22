# 加载飞桨和相关数据处理的库
from sys import modules
import paddle
from paddle.batch import batch
import paddle.fluid as fluid
from paddle.fluid.dygraph.nn import Linear
import numpy as np
import os
import gzip
import json
import random
from paddle.io import Dataset
import paddle.nn.functional as F


def load_data(mode='train'):
    # 声明数据集文件位置
    datafile = 'mnist.json.gz'
    # 加载json数据文件
    data = json.load(gzip.open(datafile))

    # 读取到的数据区分训练集，验证集，测试集
    train_set, val_set, eval_set = data
    # 数据集相关参数，图片高度IMG_ROWS, 图片宽度IMG_COLS
    IMG_ROWS = 28
    IMG_COLS = 28
    # 打印数据信息
    if mode == 'train':
        imgs, labels = train_set[0], train_set[1]
        print("训练数据集数量: ", len(imgs))
    elif mode == 'valid':
        imgs, labels = val_set[0], val_set[1]
    # print("验证数据集数量: ", len(imgs))
    elif mode == 'eval':
        imgs, labels = eval_set[0], eval_set[1]
    # print("测试数据集数量: ", len(imgs))

    # 数据校验--机器校验
    # 获得数据集长度
    imgs_length = len(imgs)
    # 断言assert，当条件为false时触发动作
    assert len(imgs)==len(labels),\
        "length of train_imgs({}) should be the same as train_labels({})".format(len(imgs),len(labels))

    # 定义序号
    index_list = list(range(imgs_length))
    # 读入数据时用到的批次大小
    BATCHSIZE = 100

    # 随机打乱训练数据的索引序号
    random.shuffle(index_list)

    # 定义数据生成器，返回批次数据
    def data_generator():
        if mode == 'train':
            # 训练模式下打乱数据
            random.shuffle(index_list)
        imgs_list = []
        labels_list = []
        for i in index_list:
            # 将数据处理成期望的格式，比如类型为float32，shape为[1, 28, 28]
            img = np.reshape(imgs[i],
                             [1, IMG_ROWS, IMG_COLS]).astype('float32')
            lable = np.reshape(labels[i], [1]).astype('float32')
            imgs_list.append(img)
            labels_list.append(lable)
            if len(imgs_list) == BATCHSIZE:
                # 获得一个batchsize的数据，并返回
                yield np.array(imgs_list), np.array(labels_list)
                # 清空数据读取列表
                imgs_list = []
                labels_list = []

        # 如果剩余数据的数目小于BATCHSIZE，
        # 则剩余数据一起构成一个大小为len(imgs_list)的mini-batch
        if len(imgs_list) > 0:
            yield np.array(imgs_list), np.array(labels_list)

    # 数据校验--人工校验
    # 声明数据读取函数，从训练集中读取数据
    train_load = data_generator
    # 以迭代的形式读取数据
    for batchh_id, data in enumerate(train_load()):
        image_data, lable_data = data
        if batchh_id == 0:
            # 打印数据shape和类型
            print("打印第一个batch数据的维度:")
            print("图像维度: {}, 标签维度: {}, 图像数据类型: {}, 标签数据类型: {}".format(
                image_data.shape, lable_data.shape, type(image_data),
                type(lable_data)))
        break

    return data_generator


# 数据异步读取
# 创建一个类MnistDataset，继承paddle.io.Dataset 这个类
# MnistDataset的作用和上面load_data()函数的作用相同，均是构建一个迭代器
class MnistDataset(paddle.io.Dataset):
    def __init__(self, mode):
        # 声明数据集文件位置
        datafile = 'mnist.json.gz'
        # 加载json数据文件
        data = json.load(gzip.open(datafile))

        # 读取到的数据区分训练集，验证集，测试集
        train_set, val_set, eval_set = data
        # 数据集相关参数，图片高度IMG_ROWS, 图片宽度IMG_COLS
        IMG_ROWS = 28
        IMG_COLS = 28
        # 打印数据信息
        if mode == 'train':
            imgs, labels = train_set[0], train_set[1]
            print("训练数据集数量: ", len(imgs))
        elif mode == 'valid':
            imgs, labels = val_set[0], val_set[1]
        # print("验证数据集数量: ", len(imgs))
        elif mode == 'eval':
            imgs, labels = eval_set[0], eval_set[1]
        # print("测试数据集数量: ", len(imgs))
        else:
            raise Exception(
                "mode can only be one of ['train', 'valid', 'eval']")

        # 数据校验--机器校验
        # 获得数据集长度
        imgs_length = len(imgs)
        # 断言assert，当条件为false时触发动作
        assert len(imgs)==len(labels),\
            "length of train_imgs({}) should be the same as train_labels({})".format(len(imgs),len(labels))

        self.imgs = imgs
        self.lables = labels

        def __getitem__(self, idx):
            img = np.array(self.imgs[idx]).astype('float32')
            lable = np.array(self.lables[idx]).astype('float32')
            return img, lable

        def __len__(self):
            return len(self.imgs)


# 数据处理部分之后的代码，数据读取的部分调用load_data函数
# 定义网络结构，同上一节所使用的网络结构
class MNIST(fluid.dygraph.Layer):
    def __init__(self):
        super(MNIST, self).__init__()
        self.fc = Linear(input_dim=784, output_dim=1, act=None)

    def forward(self, inputs):
        inputs = fluid.layers.reshape(inputs, (-1, 784))
        outputs = self.fc(inputs)
        return outputs


'''
# 训练配置，并启动训练过程
with fluid.dygraph.guard():
    model = MNIST()
    model.train()
    # 调用加载数据的函数
    train_loader = load_data('train')
    # 创建异步数据读取器
    place = fluid.CPUPlace()
    # 定义DataLoader对象用于加载Python生成器产生的数据
    data_loader = fluid.io.DataLoader.from_generator(capacity=5,
                                                     return_list=True)
    # 设置数据生成器
    data_loader.set_batch_generator(train_loader, places=place)
    # 设置优化器
    optimizer = fluid.optimizer.SGDOptimizer(learning_rate=0.001,
                                             parameter_list=model.parameters())
    EPOCH_NUM = 10
    for epoch_id in range(EPOCH_NUM):
        for batch_id, data in enumerate(train_loader()):
            # 准备数据，变得更加简洁
            image_data, label_data = data
            image = fluid.dygraph.to_variable(image_data)
            lable = fluid.dygraph.to_variable(label_data)

            # 前向计算的过程
            predict = model(image)

            # 计算损失，取一个批次样本损失的平均值
            loss = fluid.layers.square_error_cost(predict, lable)
            avg_loss = fluid.layers.mean(loss)

            # 每训练了200批次的数据，打印下当前Loss的情况
            if batch_id % 200 == 0:
                print("epoch: {}, batch: {}, loss is: {}".format(
                    epoch_id, batch_id, avg_loss.numpy()))

            # 后向传播，更新参数的过程
            avg_loss.backward()
            optimizer.minimize(avg_loss)
            model.clear_gradients()

    # 保存模型参数
    fluid.save_dygraph(model.state_dict(), 'mnist_dealdata')

同步数据读取：数据读取与模型训练串行。当模型需要数据时，才运行数据读取函数获得当前批次的
    数据。在读取数据期间，模型一直等待数据读取结束才进行训练，数据读取速度相对较慢。

异步数据读取：数据读取和模型训练并行。读取到的数据不断的放入缓存区，无需等待模型训练
    就可以启动下一轮数据读取。当模型训练完一个批次后，不用等待数据读取过程，直接从缓存区
    获得下一批次数据进行训练，从而加快了数据读取速度。

异步队列：数据读取和模型训练交互的仓库，二者均可以从仓库中读取数据，它的存在使得两者
    的工作节奏可以解耦。

# 定义数据读取后存放的位置，CPU或者GPU，这里使用CPU
# place = fluid.CUDAPlace(0) 时，数据才读取到GPU上
place = fluid.CPUPlace()
with fluid.dygraph.guard(place):
    # 声明数据加载函数，使用训练模式
    train_loader = load_data('train')
    # 定义DataLoader对象用于加载Python生成器产生的数据
    # 创建一个DataLoader对象用于加载Python生成器产生的数据。数据会由Python线程预先读取，
    # 并异步送入一个队列中。
    data_loader = fluid.io.DataLoader.from_generator(capacity=5,
                                                     return_list=True)

    fluid.io.DataLoader.from_generator参数名称和含义如下：

feed_list：仅在PaddlePaddle静态图中使用，动态图中设置为“None”，
    本教程默认使用动态图的建模方式；

capacity：表示在DataLoader中维护的队列容量，如果读取数据的速度很快，建议设置为更大的值；

use_double_buffer：是一个布尔型的参数，设置为“True”时，
    Dataloader会预先异步读取下一个batch的数据并放到缓存区；

iterable：表示创建的Dataloader对象是否是可迭代的，一般设置为“True”；

return_list：在动态图模式下需要设置为“True”

    # 设置数据生成器
    # 用创建的DataLoader对象设置一个数据生成器set_batch_generator，
    # 输入的参数是一个Python数据生成器train_loader和服务器资源类型place（标明CPU还是GPU）
    data_loader.set_batch_generator(train_loader, places=place)
    # 迭代的读取数据并打印数据的形状
    for i, data in enumerate(data_loader):
        image_data, lable_data = data
        print(i, image_data.shape, lable_data.shape)
        if i > 5:
            break
'''

# 声明数据加载函数，使用训练模式，MnistDataset构建的迭代器每次迭代只返回batch=1的数据
train_dataset = MnistDataset(mode='train')
# 使用paddle.io.DataLoader 定义DataLoader对象用于加载Python生成器产生的数据，
# DataLoader 返回的是一个批次数据迭代器，并且是异步的；
data_loader = paddle.io.DataLoader(train_dataset, batch_size=100, shuffle=True)
# 迭代的读取数据并打印数据的形状
for i, data in enumerate(data_loader()):
    imgs, lables = data
    print(i, imgs.shape, lables.shape)
    if i>2:
        break


def train(model):
    model = MNIST()
    model.train()
    opt = paddle.optimizer.SGD(learning_rate=0.001,
                               parameters=model.parameters())
    EPOCH_NUM = 10
    for epoch_id in range(EPOCH_NUM):
        for batch_id, data in enumerate(data_loader()):
            images, lables = data
            images = paddle.to_tensor(images)
            lables = paddle.to_tensor(lables).astype('float32')

            #前向计算的过程
            predicts = model(images)

            #计算损失，取一个批次样本损失的平均值
            loss = F.square_error_cost(predicts, lables)
            avg_loss = paddle.mean(loss)

            # 每训练了200批次的数据，打印下当前Loss的情况
            if batch_id % 200 == 0:
                print("epoch: {}, batch: {}, loss is: {}".format(
                    epoch_id, batch_id, avg_loss.numpy()))

            # 后向传播，更新参数的过程
            avg_loss.backward()
            opt.step()
            opt.clear_grad()

    # 保存模型参数
    paddle.save(model.state_dict(), 'mnist_dealdata_yibu')


# 创建模型
model = MNIST()
# 启动训练过程
train(model)