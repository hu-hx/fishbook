import numpy as np


###  损失函数  ###

# 均方误差
def mean_squared_error(y,t):
    return 0.5*np.sum((y-t)**2)

# 交叉熵误差
def cross_entropy_error(y,t,one_hot):
    if y.ndim == 1:
        t = t.reshape(1,t.size)
        y = y.reshape(1,y.size)
    batch_size = y.shape[0]

    if one_hot:
        return -np.sum( t * np.log(y + 1e-7) ) / batch_size     # one_hot

    return -np.sum( np.log(y[np.arange(batch_size) , t] + 1e-7) ) / batch_size



###  激活函数  ###

# sigmoid: 一般用于二分类输出层
def sigmoid(x):
    return 1/(1+np.exp(-x))

# softmax:一般用于多元分类输出层
def softmax(x):
    if x.ndim == 2:
        x = x - np.max(x, axis=1, keepdims=True)  # 数值稳定
        exp_x = np.exp(x)
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    x = x - np.max(x)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x)


#identity_function: 一般用于回归问题输出层
def identity(x):
    return x



###  数值微分  ###

def numerical_gradient(f,x):
    dx = 1e-4
    grad = np.zeros_like(x)

    # 提供遍历坐标的迭代器
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index

        tmp_val = x[idx]
        # f(x + dx):
        x[idx] = tmp_val + dx
        fx1 = f(x)

        #f(x - dx):
        x[idx] = tmp_val - dx
        fx2 = f(x)

        grad[idx] = (fx1 - fx2) / (2*dx)
        x[idx] = tmp_val # 还原

        it.iternext() # 迭代器前进

    return grad


def im2col(input_data, filter_h, filter_w, stride=1, pad=0):
    """

    Parameters
    ----------
    input_data : 由(数据量, 通道, 高, 长)的4维数组构成的输入数据
    filter_h : 滤波器的高
    filter_w : 滤波器的长
    stride : 步幅
    pad : 填充

    Returns
    -------
    col : 2维数组
    """
    N, C, H, W = input_data.shape
    out_h = (H + 2*pad - filter_h)//stride + 1
    out_w = (W + 2*pad - filter_w)//stride + 1

    img = np.pad(input_data, [(0,0), (0,0), (pad, pad), (pad, pad)], 'constant')
    col = np.zeros((N, C, filter_h, filter_w, out_h, out_w))

    for y in range(filter_h):
        y_max = y + stride*out_h
        for x in range(filter_w):
            x_max = x + stride*out_w
            col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]

    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N*out_h*out_w, -1)
    return col

def col2im(col, input_shape, filter_h, filter_w, stride=1, pad=0):
    """

    Parameters
    ----------
    col :
    input_shape : 输入数据的形状（例：(10, 1, 28, 28)）
    filter_h :
    filter_w
    stride
    pad

    Returns
    -------

    """
    N, C, H, W = input_shape
    out_h = (H + 2*pad - filter_h)//stride + 1
    out_w = (W + 2*pad - filter_w)//stride + 1
    col = col.reshape(N, out_h, out_w, C, filter_h, filter_w).transpose(0, 3, 4, 5, 1, 2)

    img = np.zeros((N, C, H + 2*pad + stride - 1, W + 2*pad + stride - 1))
    for y in range(filter_h):
        y_max = y + stride*out_h
        for x in range(filter_w):
            x_max = x + stride*out_w
            img[:, :, y:y_max:stride, x:x_max:stride] += col[:, :, y, x, :, :]

    return img[:, :, pad:H + pad, pad:W + pad]













