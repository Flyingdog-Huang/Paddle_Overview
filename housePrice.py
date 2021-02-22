import numpy as np
import json
from matplotlib import pyplot as plt


# 封装成load data函数
def load_data():
    # 读入训练数据
    datafile = 'housing.data'
    data = np.fromfile(datafile, sep=' ')
    # print(data)

    # 读入之后的数据被转化成1维array，其中array的第0-13项是第一条数据，第14-27项是第二条数据，以此类推....
    # 这里对原始数据做reshape，变成N x 14的形式
    feature_name = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS',
                    'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
    feature_num = len(feature_name)
    data = data.reshape([data.shape[0] // feature_num, feature_num])

    # 查看数据
    # x = data[0]
    # print(x.shape)
    # print(x)

    ratio = 0.8
    offset = int(data.shape[0] * ratio)
    training_data = data[:offset]
    # print(training_data.shape)

    # 计算train数据集的最大值，最小值，平均值
    maximums, minimums, avgs = \
        training_data.max(axis=0), \
        training_data.min(axis=0), \
        training_data.sum(axis=0) / training_data.shape[0]

    # 对数据进行归一化处理
    for i in range(feature_num):
        data[:, i] = (data[:, i] - minimums[i]) / (maximums[i] - minimums[i])

    # 训练集和测试集的划分比例
    training_data = data[:offset]
    test_data = data[offset:]
    return training_data, test_data


# 获取数据
training_data, test_data = load_data()


# x = training_data[:, :-1]
# y = training_data[:, -1:]


# 查看数据
# print(x[0])
# print(y[0])

# 模型设计
class Network(object):
    def __init__(self, num_of_weights):
        # 随机产生W的初始值
        # 为了保持程序每次运行结果的一致性，
        # 此处设置固定的随机数种子
        np.random.seed(0)
        self.w = np.random.randn(num_of_weights, 1)
        self.b = 0.

    def forward(self, x):
        '''
        正向计算
        :param x:
        :return:
        '''
        z = np.dot(x, self.w) + self.b
        return z

    def loss(self, z, y):
        '''
        误差计算
        :param z:
        :param y:
        :return:
        '''
        error = z - y
        cost = error ** 2
        cost = np.mean(cost)
        return cost

    def gradient(self, x, y):
        '''
        梯度计算
        :param x:
        :param y:
        :return:
        '''
        z = self.forward(x)
        gradient_w = (z - y) * x
        gradient_w = np.mean(gradient_w, axis=0)
        gradient_w = gradient_w[:, np.newaxis]
        gradient_b = (z - y)
        gradient_b = np.mean(gradient_b)
        return gradient_w, gradient_b

    def update(self, gradient_w, gradient_b, eta=0.01):
        '''
        参数更新
        :param gradient_w:
        :param gradient_b:
        :param eta:
        :return:
        '''
        self.w = self.w - eta * gradient_w
        self.b = self.b - eta * gradient_b

    def train(self, training_data, num_epochs, batch_size=10, eta=0.01):
        '''
        开始训练(随机梯度下降)
        :param x:
        :param y:
        :param iterations:
        :param eta:
        :return:
        '''
        losses = []
        n = len(training_data)
        for epoch_id in range(num_epochs):
            # 在每轮迭代开始之前，将训练数据的顺序随机打乱
            # 然后再按每次取batch_size条数据的方式取出
            np.random.shuffle(training_data)
            # 将训练数据进行拆分，每个mini_batch包含batch_size条的数据
            mini_batches = [training_data[k:k + batch_size] for k in range(0, n, batch_size)]
            for iter_id, mini_batch in enumerate(mini_batches):
                x = mini_batch[:, :-1]
                y = mini_batch[:, -1:]
                z = self.forward(x)
                L = self.loss(z, y)
                gradient_w, gradient_b = self.gradient(x, y)
                self.update(gradient_w, gradient_b, eta)
                losses.append(L)
                print('epoch {:3d} / iter {:3d}, loss = {:.4f}'.format(epoch_id, iter_id, L))
        return losses


# 创建网络
net = Network(13)
# num_iterations = 1000
# 启动训练
losses = net.train(training_data, num_epochs=50, batch_size=100, eta=0.1)

# 画出损失函数变化趋势
plot_x = np.arange(len(losses))
plot_y = np.array(losses)
plt.plot(plot_x, plot_y)
plt.show()
