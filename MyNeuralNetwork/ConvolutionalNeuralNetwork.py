from .layers import *
from .optimizer import *

from collections import OrderedDict
import matplotlib.pyplot as plt
import pickle
import time


"""
input_dim=(1, 28, 28), conv_param={'filter_num':30, 'filter_size':5, 'pad':0, 'stride':1}
"""

class ConvolutionalNeuralNetwork:
    def __init__(self, input_dim,conv_param:dict, hidden_size_list, output_size,conv_pool_num=1,acativation_function ='Relu', BatchNorm = False, dropout = False, dropout_ration = 0.5, one_hot = False):
        # input_dim = (C:通道数,H：高度,W：宽度) (1, 28, 28)
        # conv_param={'filter_num':30, 'filter_size':5, 'pad':0, 'stride':1} '滤波器数量' '滤波器尺寸(当成方形)' '填充' '步幅'

        filter_num = conv_param['filter_num']
        filter_size = conv_param['filter_size']
        pad = conv_param['pad']
        stride = conv_param['stride']
        pool_size = 2



        # L2正则化 权值衰减系数
        self.weight_decay_lambda = 0
        # 是否加入BatchNorm
        self.BatchNorm = BatchNorm
        # 是否加入dropout
        self.dropout = dropout
        # 防止内存超限设置的最大计算数量
        self.max_calcu_num = None

        # 创建对应的激活函数和对应的标准差系数
        try:
            acativation_function_layer = globals()[acativation_function + "Layer"]()
            weight_init_std_coff = 2 if acativation_function == "Relu" else 1    # 这里要确保仅有这两种激活函数

        except KeyError:
            print(f"不存在激活函数:{acativation_function},已使用Relu\n")
            weight_init_std_coff = 2


        # 初始化权重和层
        self.params = {}
        self.layers = OrderedDict()

        ###  构建卷积池化层部分  ###
        # 卷积池化部分数量
        self.k = conv_pool_num

        # 卷积层权重
        weight_init_std = 0.01
        pool_output_size = input_dim[1]  # 这样设置方便后边计算多层卷积层
        C = input_dim[0]

        for i in range(1,self.k+1):
            W_i, b_i = f"W{i}", f"b{i}"
            Relu_i = f"Relu{i}"
            Convolution_i,pool_i = f"Convolution{i}", f"pool{i}"

            conv_output_size = (pool_output_size + 2 * pad - filter_size) / stride + 1
            pool_output_size = conv_output_size / pool_size


            self.params[W_i] = weight_init_std * np.random.randn(filter_num,C,filter_size,filter_size)
            self.params[b_i] =  np.zeros(filter_num)

            # 构建卷积-Relu-池化层
            self.layers[Convolution_i] = Convolution(self.params[W_i],self.params[b_i],stride,pad)
            self.layers[Relu_i]  =  ReluLayer()
            self.layers[pool_i] = Pooling( pool_size, pool_size, stride = 2)

            # 通道数更新为上一层滤波器数量
            C = filter_num

        pool_output_num = int(filter_num * pool_output_size**2)




        ###  构建全连接层部分  ###

        # 储存全连接层需要用到的层数尺寸
        all_layers_size =  [pool_output_num] + hidden_size_list + [output_size]

        # 计算非卷积池化网络层数
        self.n = len(hidden_size_list) + 1

        for i in range(1 + self.k, self.n + self.k + 1):

            # 设置字符
            W_i, b_i = f"W{i}", f"b{i}"                               # 权重, 偏置
            Affine_i, Acative_i = f"Affine{i}", f"Acative{i}"         # 传播层，激活函数层
            BatchNorm_i,Dropout_i = f"BatchNorm{i}",f"Dropout{i}"     # BatchNorm 层
            gamma_i, beta_i = f"gamma{i}", f"beta{i}"                 # BatchNorm 层系数

            # 根据Relu激活函数初始化权重的标准差
            weight_init_std = np.sqrt(weight_init_std_coff / all_layers_size[i-1-self.k])

            # 初始化权重参数
            self.params[W_i] = weight_init_std * np.random.randn(all_layers_size[i-1-self.k],all_layers_size[i-self.k])
            self.params[b_i] = np.zeros(all_layers_size[i-self.k])

            # 构建 Affine + BatchNorm + Relu + Dropout 层
            if i != self.n + self.k:
                # Affine: 普通传播层
                self.layers[Affine_i] = Affine(self.params[W_i], self.params[b_i])

                if BatchNorm:
                    # BatchNorm: 批量正规化
                    self.params[gamma_i] = np.ones(hidden_size_list[i - 1 - self.k])
                    self.params[beta_i] = np.zeros(hidden_size_list[i - 1 - self.k])
                    self.layers[BatchNorm_i] = BatchNormalization(self.params[gamma_i], self.params[beta_i])

                # 激活函数层
                self.layers[Acative_i] = globals()[acativation_function + "Layer"]()

                if dropout:
                    # Dropout: 随机让神经元失效
                    self.layers[Dropout_i] = Dropout(dropout_ration)


            else:  # 输出层
                self.layers[Affine_i] = Affine(self.params[W_i], self.params[b_i])
                self.lastLayer = SoftmaxWithLoss(one_hot)




    # 预测值：网络不经过输出层的输出
    def predict(self,x,train_flg=True):
        for key,layer in self.layers.items():
            if "Dropout" in key or "BatchNorm" in key:
                x = layer.forward(x,train_flg)
            else:
                x = layer.forward(x)
        return x

    # x:输入数据 t:监督数据
    def loss(self,x,t):
        y = self.predict(x,train_flg=True)
        # 输出层的前向传播是交叉熵误差
        return self.lastLayer.forward(y,t)

    # 计算准确率
    def accuracy(self, x, t):
        batch_size = self.max_calcu_num  # 设置每次预测最大样本数目，防止直接把整个train_set放进去内存超限

        y = self.predict(x[:min(batch_size, x.shape[0])], train_flg=False)
        y = np.argmax(y, axis=1)

        if t.ndim != 1:  # one_hot
            t = np.argmax(t, axis=1)

        correct = np.sum(y == t[:y.shape[0]])

        for i in range(batch_size, x.shape[0], batch_size):
            xb = x[i:i + batch_size]
            yb = self.predict(xb, train_flg=False)
            yb = np.argmax(yb, axis=1)
            correct += np.sum(yb == t[i:i + yb.shape[0]])

        accuracy = correct / float(x.shape[0])
        return accuracy

    # 数值微分求梯度
    def numerical_gradient(self, x, t):
        loss_W = lambda W: self.loss(x, t)
        # 这里的W是形式函数，实际不传入，而权重会在下面的numerical_gradient中修改

        grads = {}
        for key in self.params.keys():
            grads[key] = numerical_gradient(loss_W, self.params[key])
            if "W" in key:
                if int(key[1]) <= self.k:
                    grads[key] += self.weight_decay_lambda * self.layers["Convolution"+key[1]].W
                else:
                    grads[key] += self.weight_decay_lambda * self.layers["Affine" + key[1]].W

        return grads

    # 反向传播求梯度
    def backward_gradient(self,x,t):
         # forward
        self.loss(x, t)

       # backward
        dout = 1
        dout = self.lastLayer.backward(dout)
        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        grads = {}

        # 卷积层梯度
        for i in range(1, self.k + 1):
            W_i, b_i = f"W{i}", f"b{i}"
            Convolution_i = f"Convolution{i}"
            grads[W_i] = self.layers[Convolution_i].dW
            grads[b_i] = self.layers[Convolution_i].db

        # 全连接层梯度
        for i in range(1 + self.k, self.n + self.k + 1):
            W_i, b_i, Affine_i = f"W{i}", f"b{i}", f"Affine{i}"
            BatchNorm_i = f"BatchNorm{i}"
            gamma_i, beta_i = f"gamma{i}", f"beta{i}"

            grads[W_i] = self.layers[Affine_i].dW + self.weight_decay_lambda * self.layers[Affine_i].W
            grads[b_i] = self.layers[Affine_i].db

            if i != (self.n + self.k) and self.BatchNorm:
                grads[gamma_i] = self.layers[BatchNorm_i].dgamma
                grads[beta_i] = self.layers[BatchNorm_i].dbeta

        return grads

    # 梯度确认
    def gradient_check(self,x_check,t_check):

        if self.dropout:
            print("开启dropout不能梯度确认")
            return

        grad_numerical = self.numerical_gradient(x_check, t_check)
        grad_backprop = self.backward_gradient(x_check, t_check)
        print("数值微分和反向传播计算权重绝对误差的平均值")
        for key in grad_numerical.keys():
            diff = np.average(np.abs(grad_backprop[key] - grad_numerical[key]))
            print(key + ":" + str(format(diff,'.0e')))
        print("\n")




    # 训练函数
    def train(self,epcohs,batch_size,dataset,optimizer = 'SGD',weight_decay_lambda = 0,max_calcu_num = 10000):

        x_train, t_train, x_test, t_test = dataset[0],dataset[1],dataset[2],dataset[3]

        self.weight_decay_lambda = weight_decay_lambda
        self.max_calcu_num = max_calcu_num

        # 训练集尺寸
        train_size = x_train.shape[0]
        # 迭代次数
        iters_num = int(train_size * epcohs / batch_size) + 1

        # 记录损失值和正确率
        train_loss_list = []
        train_acc_list = []
        test_acc_list = []

        # 创建对应的优化器
        try:
            optimizer_ = globals()[optimizer]()
        except KeyError:
            print(f"不存在优化器:{optimizer},已使用SGD优化器\n")
            optimizer_ = SGD()


        # 每个epoch要算几批
        iter_per_epoch = max(train_size // batch_size, 1)
        epochs = 0

        t_epochs = time.perf_counter()
        for t in range(iters_num):

            # 每epoch计算一次准确率
            if t % iter_per_epoch == 0:
                t1 = time.perf_counter()
                train_acc = self.accuracy(x_train, t_train)
                t2 = time.perf_counter()
                test_acc = self.accuracy(x_test, t_test)
                t3 = time.perf_counter()
                print(f"\nepochs{epochs}:")
                print(f"训练集预测耗时{t2-t1:.2f}s")
                print(f"测试集预测耗时{t3-t2:.2f}s")

                train_acc_list.append(train_acc)
                test_acc_list.append(test_acc)
                print(f"train_acc： {train_acc:.3f}")
                print(f"test_acc： {test_acc:.3f}")
                print(f"总耗时 {time.perf_counter() - t_epochs:.2f}s\n")
                t_epochs = time.perf_counter()
                epochs+=1

            batch_mask = np.random.choice(train_size, batch_size)
            x_batch = x_train[batch_mask]
            t_batch = t_train[batch_mask]

            # 梯度
            #grad = self.numerical_gradient(x_batch, t_batch)
            grad = self.backward_gradient(x_batch, t_batch)

            # 更新
            optimizer_.update(self.params,grad)

            loss = self.loss(x_batch, t_batch)
            train_loss_list.append(loss)
            # print(f"\r批次{t}/{iters_num}计算耗时：{t_end-t_start:.2f}s".ljust(60), end="") 为什么jupyter打印不了这个



        print("\ntest_accuracy: ", end="")
        print(test_acc_list[-1])
        self.plot(iters_num,train_loss_list,train_acc_list,test_acc_list)




    @staticmethod
    def plot(iters_num, train_loss_list,train_acc_list,test_acc_list):

        ### 绘制损失函数图像 ###

        plt.figure(figsize=(8, 5))

        # 绘制损失曲线
        plt.plot(range(iters_num), train_loss_list,linewidth=2, color='b', label='Training Loss')
        # 设置坐标轴标签
        plt.xlabel('Iteration', fontsize=14)
        plt.ylabel('Loss', fontsize=14)
        # 设置标题
        plt.title('Training Loss Curve', fontsize=16, fontweight='bold')
        # 添加网格（可选）
        plt.grid(True, alpha=0.3, linestyle='--')
        # 添加图例
        plt.legend(fontsize=15)

        # 如果损失下降很快，可以使用对数坐标
        # if train_loss_list[0] / train_loss_list[-1] > 100:  # 损失下降超过100倍
        #     plt.yscale('log')
        #     plt.title('Training Loss Curve (Log Scale)', fontsize=14, fontweight='bold')

        # 自动调整布局
        plt.tight_layout()
        # 显示图像
        plt.show()


        ### 绘制正确率函数图像 ###

        plt.figure(figsize=(8, 5))
        epochs_num = len(train_acc_list)
        plt.plot(range(epochs_num), train_acc_list, linewidth=2, color='b', label='Training Accuracy')
        plt.plot(range(epochs_num), test_acc_list, linewidth=2, color='r', label='Test Accuracy')

        # 设置坐标轴标签
        plt.xlabel('epochs', fontsize=14)
        plt.ylabel('accuracy', fontsize=14)
        # 设置标题
        plt.title('accuaacy with train', fontsize=16, fontweight='bold')
        # 添加网格（可选）
        plt.grid(True, alpha=0.3, linestyle='--')
        # 添加图例
        plt.legend(fontsize=15)

        # 自动调整布局
        plt.tight_layout()
        # 显示图像
        plt.show()


    # 保存参数
    def save(self, path="./params.pkl"):
        payload = {
            "params": self.params,
            }
        with open(path, "wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)

        print(f"模型参数已保存在{path}")

    # 加载参数
    def load(self, path="./params.pkl"):
        with open(path, "rb") as f:
            payload = pickle.load(f)


        if 'params' not in payload:
            print("加载失败：参数文件中没有'params'")
            return

        params = payload['params']

        # key 结构不一致，直接失败
        if set(params.keys()) != set(self.params.keys()):
            print("加载失败：'params'结构不一致")
            return


        self.params = params

        for i in range(1, self.n + 1):
            affine = self.layers.get(f"Affine{i}")
            if affine is None:
                continue
            affine.W = self.params[f"W{i}"]
            affine.b = self.params[f"b{i}"]

        print(f"成功从{path}中加载模型参数")

