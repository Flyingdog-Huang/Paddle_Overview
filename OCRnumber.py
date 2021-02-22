from paddle.fluid.layers import tensor
from rewriteHousePriceModleByPaddle import BATCH_SIZE, EPOCH_NUM
from numpy.lib.type_check import imag
import paddle
import paddle.fluid as fluid
from paddle.fluid.dygraph.nn import Linear
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
import gzip
import json
import random

# 如果～/.cache/paddle/dataset/mnist/目录下没有MNIST数据，API会自动将MINST数据下载到该文件夹下
# 设置数据读取器，读取MNIST数据训练集
trainset = paddle.dataset.mnist.train()
# 包装数据读取器，每次读取的数据数量设置为batch_size=8
train_reader = paddle.batch(trainset, batch_size=8)

# global img_data
# global lable_data

# 以迭代的形式读取数据
for batch_id, data in enumerate(train_reader()):
    # 获取图像数据，并转为float32类型的数组
    img_data = np.array([x[0] for x in data]).astype('float32')
    # 获取数据标签，并转为float32类型的数组
    lable_data = np.array([x[1] for x in data]).astype('float32')
    # 打印数据形状
    # print("图像数据形状和对应数据为：", img_data.shape, img_data[0])
    # print("图像标签形状和对应数据为：", lable_data.shape, lable_data[0])
    break

# print("\n打印第一个batch的第一个图像，对应标签数字为{}".format(lable_data[0]))
# img = np.array(img_data[0] + 1) * 127.5
# img = np.reshape(img, [28, 28]).astype(np.uint8)

# plt.figure("image")  # 图像窗口名称
# plt.imshow(img)
# plt.axis('on')  # 关掉坐标轴
# plt.title('img')  # 图像名字
# plt.show()


# 定义mnist数据识别网络结构，同房价预测网络
class MNIST(fluid.dygraph.Layer):
    def __init__(self):
        super(MNIST, self).__init__()

        # 定义一层全连接层，输出维度是1，激活函数为None
        self.fc = Linear(input_dim=784, output_dim=1, act=None)

    # 定义网络结构的前向计算过程
    def forward(self, inputs):
        outputs = self.fc(inputs)
        return outputs


# 定义工作环境
with fluid.dygraph.guard():
    # 声明网络结构
    model = MNIST()
    # 启动训练模式
    model.train()
    # 定义数据读取函数，数据读取batch_size设置为16
    train_loader = paddle.batch(paddle.dataset.mnist.train(), batch_size=16)
    # 定义优化器，使用随机梯度下降SGD优化器，学习率设置为0.001
    optimizer = fluid.optimizer.SGDOptimizer(learning_rate=0.001,
                                             parameter_list=model.parameters())

# 通过with语句创建一个dygraph运行的context
# 动态图下的一些操作需要在guard下进行
with fluid.dygraph.guard():
    model = MNIST()
    model.train()
    train_loader = paddle.batch(paddle.dataset.mnist.train(), batch_size=16)
    optimizer = fluid.optimizer.SGDOptimizer(learning_rate=0.001,
                                             parameter_list=model.parameters())
    EPOCH_NUM = 10
    for epoch_id in range(EPOCH_NUM):
        for batch_id, data in enumerate(train_loader()):
            # 准备数据，格式需要符合框架要求
            image_data = np.array([x[0] for x in data]).astype('float32')
            lable_data = np.array([x[1] for x in data
                                   ]).astype('float32').reshape(-1, 1)
            # 将数据转为支持的标准格式
            image = fluid.dygraph.to_variable(image_data)
            lable = fluid.dygraph.to_variable(lable_data)

            # 前向计算
            predict = model(image)

            # 计算损失
            loss = fluid.layers.square_error_cost(predict, lable)
            avg_loss = fluid.layers.mean(loss)

            # 打印训练结果
            if batch_id != 0 and batch_id % 1000 == 0:
                print("epoch: {}, batch: {}, loss is : {}".format(
                    epoch_id, batch_id, avg_loss.numpy()))

            # 后向传播
            avg_loss.backward()
            optimizer.minimize(avg_loss)
            model.clear_gradients()

    # 保存模型
    fluid.save_dygraph(model.state_dict(), 'mnist')

# 模型测试
'''
模型测试的主要目的是验证训练好的模型是否能正确识别出数字，包括如下四步：

1`声明实例
2`加载模型：加载训练过程中保存的模型参数。
3`灌入数据：将测试样本传入模型，模型的状态设置为校验状态（eval），显式告诉框架我们接下来只会使用前向计算的流程，不会计算梯度和梯度反向传播。
4`获取预测结果，取整后作为预测标签输出。
在模型测试之前，需要先从'./work/example_0.jpg'文件中读取样例图片，并进行归一化处理。
'''


# 读取一张本地图片并转换成标准格式
def load_image(img_path):
    # 从img_path中读取图像，并转为灰度图
    im = Image.open(img_path).convert('L')
    print(np.array(im))
    im = im.resize((28, 28), Image.ANTIALIAS)
    im = np.array(im).reshape(1, -1).astype(np.float32)
    # 图像归一化
    im = 1 - im / 127.5
    return im


# 定义预测过程
with fluid.dygraph.guard():
    model = MNIST()
    params_file_path = 'mnist'
    img_path = 'example_0.png'
    # 加载模型参数
    model_dict, _ = fluid.load_dygraph("mnist")
    model.load_dict(model_dict)
    # 灌入数据
    model.eval()
    tensor_img = load_image(img_path)
    result = model(fluid.dygraph.to_variable(tensor_img))
    #  预测输出取整，即为预测的数字，打印结果
    print("本次预测的数字是", result.numpy().astype('int32'))
