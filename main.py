import numpy as np

from libsvm.svmutil import *

from sklearn import datasets
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams['font.sans-serif'] = ['KaiTi']
matplotlib.rcParams['axes.unicode_minus'] = False


def sigmoid(x):
    return np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)


class BP_NN:

    def __init__(self, train_data_, test_data_, layer_num, node_nums, lr, epochs):

        self.train_data = train_data_
        self.test_data = test_data_
        self.N, self.M = train_data_["features"].shape
        self.C = train_data_["labels"].shape[1]
        self.N_test = test_data_["labels"].shape[0]

        self.layer_num = layer_num
        self.node_nums = node_nums
        self.lr = lr
        self.epochs = epochs

        self.train_prediction = None
        self.test_prediction = None

        # 初始化各边的权重，设置为符合正态分布的随机值
        self.weight = [np.random.normal(0.0, 1, (self.M, node_nums[0]))]
        for j in range(self.layer_num - 1):
            self.weight.append(np.random.normal(0.0, 1, (self.node_nums[j], self.node_nums[j + 1])))
        self.weight.append(np.random.normal(0.0, 1, (node_nums[-1], self.C)))

        # 记录损失值和准确率变化，便于可视化
        self.loss_list = []
        self.acc_list = []

        self.train()
        self.visualize()

    def loss(self):
        return 0.5 * np.sum(
            (self.train_prediction - self.train_data["labels"]) * (self.train_prediction - self.train_data["labels"]))

    def accuracy(self):
        predict = np.argmax(self.test_prediction, axis=-1)
        true = np.argmax(self.test_data["labels"], axis=-1)
        return np.sum(predict == true) / self.N_test

    def train(self):
        for k in range(self.epochs):

            # 前向传播
            z = [np.dot(self.train_data["features"], self.weight[0])]
            a = [sigmoid(z[0])]
            for j in range(self.layer_num):
                z.append(np.dot(a[j], self.weight[j + 1]))
                a.append(sigmoid(z[j + 1]))

            # 记录损失值
            self.train_prediction = a[-1]
            self.loss_list.append(self.loss())

            # 计算各层误差
            errors = [(self.train_data["labels"] - a[-1]) * a[-1] * a[-1]]
            for j in range(self.layer_num):
                errors.append(np.dot(errors[j], self.weight[-j - 1].T) * a[-j - 2] * (1 - a[-j - 2]))

            # 反向传播，更新权重
            for j in range(self.layer_num):
                self.weight[self.layer_num - j] -= self.lr * np.dot(a[self.layer_num - j - 1].T, -errors[j])
            self.weight[0] -= self.lr * np.dot(self.train_data["features"].T, -errors[-1])

            # 对测试集进行预测
            z = [np.dot(self.test_data["features"], self.weight[0])]
            a = [sigmoid(z[0])]
            for j in range(self.layer_num):
                z.append(np.dot(a[j], self.weight[j + 1]))
                a.append(sigmoid(z[j + 1]))

            # 记录准确率
            self.test_prediction = a[-1]
            self.acc_list.append(self.accuracy())
            print("第", k + 1, "次训练集损失值：", self.loss(), "对测试集预测准确率：", self.accuracy())
        print("迭代", self.epochs, "次后对测试集预测准确率达到：", self.accuracy())

    def visualize(self):
        fig, axes = plt.subplots(1, 2)
        fig.suptitle("BP神经网络对iris数据分类")
        axes[0].set_xlabel("迭代数")
        axes[0].set_ylabel("损失值")
        axes[0].set_title("损失值变化图")
        axes[1].set_xlabel("迭代数")
        axes[1].set_ylabel("准确率")
        axes[1].set_title("准确率变化图")
        epochs_list = np.arange(self.epochs)
        axes[0].plot(epochs_list, self.loss_list, color=(0, 0, 0))
        axes[1].plot(epochs_list, self.acc_list, color=(0, 0, 0))
        plt.show()


def using_BPNN(train_data_, test_data_):
    print("输入隐藏层数：")
    Layer_Num = int(input())
    Node_Num = []
    for i in range(Layer_Num):
        print("输入第", i + 1, "层节点数：")
        Node_Num.append(int(input()))
    print("输入学习率：")
    Lr = float(input())
    print("输入迭代数：")
    Epochs = int(input())

    bp = BP_NN(train_data, test_data, Layer_Num, Node_Num, Lr, Epochs)


def using_libsvm(train_data_,test_data_):
    train_labels = np.argmax(train_data_["labels"], axis=-1)
    test_labels =np.argmax(test_data["labels"], axis=-1)
    lina_options = '-t 0 -c 1 '  # 线性核
    model = svm_train(train_labels, train_data_["labels"], lina_options)
    svm_predict(test_labels, test_data_["labels"], model)


if __name__ == '__main__':
    # 调用iris数据集
    iris = datasets.load_iris()
    # 使用包的方法将特征参数标准化
    data = preprocessing.normalize(iris.get('data'))
    # 使用包的方法将标签数据转化为独热码
    target = OneHotEncoder().fit_transform(iris.get('target').reshape(iris.get('target').shape[0], 1)).toarray()
    # 使用包的方法将数据集划分为数据集和测试集
    train_data, test_data = {}, {}
    train_data["features"], test_data["features"], train_data["labels"], test_data["labels"] = train_test_split(data, target)

    using_BPNN(train_data, test_data)
    using_libsvm(train_data, test_data)
