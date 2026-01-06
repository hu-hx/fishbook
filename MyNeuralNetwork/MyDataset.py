import torchvision
import numpy as np
import os
from torch.utils.data import Dataset
from PIL import Image
from pathlib import Path

base = str(Path(__file__).resolve().parent)   # 当前这个 .py 文件所在文件夹



def load_mnist():
    # 读入数据
    train_set = torchvision.datasets.MNIST(root = (base+"/dataset"), train=True, download=True)
    test_set = torchvision.datasets.MNIST(root = (base+"/dataset"), train=False, download=True)
    x_train, t_train, x_test, t_test = train_set.data, train_set.targets, test_set.data, test_set.targets

    # 归一化 和 转为numpy数组
    x_train = x_train.reshape(x_train.shape[0], 1, x_train.shape[1], x_train.shape[2]) / 255
    x_test = x_test.reshape(x_test.shape[0], 1, x_test.shape[1],x_test.shape[2]) / 255
    x_train, x_test, t_train, t_test = x_train.numpy(), x_test.numpy(), t_train.numpy(), t_test.numpy()

    x_train = x_train.astype(np.float16)
    x_test = x_test.astype(np.float16)

    return x_train,x_test,t_train,t_test





