from .layers import *
from .optimizer import *

from collections import OrderedDict
import matplotlib.pyplot as plt
import pickle

class DeepNeuralNetwork:
    def __init__(self, input_size, hidden_size_list, output_size, acativation_function ='Relu', BatchNorm = False, dropout = False, dropout_ration = 0.5, one_hot = False):

        # L2正则化 权值衰减系数
        self.weight_decay_lambda = 0
        # 是否加入BatchNorm
        self.BatchNorm = BatchNorm
        # 是否加入dropout
        self.dropout = dropout

        # 创建对应的激活函数和对应的标准差系数
        try:
            acativation_function_layer = globals()[acativation_function + "Layer"]()
            weight_init_std_coff = 2 if acativation_function == "Relu" else 1    # 这里要确保仅有这两种激活函数

        except KeyError:
            print(f"不存在激活函数:{acativation_function},已使用Relu\n")
            acativation_function = 'Relu'
            weight_init_std_coff = 2

        # 计算网络层数
        self.n = len(hidden_size_list) + 1

        # 初始化权重和层
        self.params = {}
        self.layers = OrderedDict()

        # 储存所有层数大小
        all_layers_size = [input_size] + hidden_size_list + [output_size]

        for i in range(1,self.n+1):

            # 设置字符
            W_i, b_i = f"W{i}", f"b{i}"                               # 权重, 偏置
            Affine_i, Acative_i = f"Affine{i}", f"Acative{i}"         # 传播层，激活函数层
            BatchNorm_i,Dropout_i = f"BatchNorm{i}",f"Dropout{i}"     # BatchNorm 层
            gamma_i, beta_i = f"gamma{i}", f"beta{i}"                 # BatchNorm 层系数

            # 根据Relu激活函数初始化权重的标准差
            weight_init_std = np.sqrt(weight_init_std_coff / all_layers_size[i-1])

            # 初始化权重参数
            self.params[W_i] = weight_init_std * np.random.randn(all_layers_size[i-1],all_layers_size[i])
            self.params[b_i] = np.zeros(all_layers_size[i])

            # 建立层
            if i != self.n:  # Affine + BatchNorm + Relu + Dropout
                # Affine: 普通传播层
                self.layers[Affine_i] = Affine(self.params[W_i], self.params[b_i])

                if BatchNorm:
                    # BatchNorm: 批量正规化
                    self.params[gamma_i] = np.ones(hidden_size_list[i - 1])
                    self.params[beta_i] = np.zeros(hidden_size_list[i - 1])
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
    def accuracy(self,x,t):
        y = self.predict(x,train_flg=False)
        y = np.argmax(y,axis=1) # 找到每行最大预测值的下标
        if t.ndim != 1: # one_hot
            t = np.argmax(t,axis=1)
        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy

    # 数值微分求梯度
    def numerical_gradient(self, x, t):
        loss_W = lambda W: self.loss(x, t)
        # 这里的W是形式函数，实际不传入，而权重会在下面的numerical_gradient中修改

        grads = {}
        for key in self.params.keys():
            grads[key] = numerical_gradient(loss_W, self.params[key])
            if "W" in key:
                grads[key] += self.weight_decay_lambda * self.layers["Affine"+key[1]].W

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
        for i in range(1, self.n + 1):
            W_i, b_i, Affine_i = f"W{i}", f"b{i}", f"Affine{i}"
            BatchNorm_i = f"BatchNorm{i}"
            gamma_i, beta_i = f"gamma{i}", f"beta{i}"

            grads[W_i] = self.layers[Affine_i].dW + self.weight_decay_lambda * self.layers[Affine_i].W
            grads[b_i] = self.layers[Affine_i].db

            if i != self.n and self.BatchNorm:
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
    def train(self,epcohs,batch_size,dataset,optimizer = 'SGD',weight_decay_lambda = 0.1):

        x_train, t_train, x_test, t_test = dataset[0],dataset[1],dataset[2],dataset[3]

        self.weight_decay_lambda = weight_decay_lambda

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
        iter_per_epoch = max(train_size / batch_size, 1)


        for t in range(iters_num):
            # 每epoch计算一次准确率,第一次为未训练
            if t % iter_per_epoch == 0:
                train_acc = self.accuracy(x_train, t_train)
                test_acc = self.accuracy(x_test, t_test)

                train_acc_list.append(train_acc)
                test_acc_list.append(test_acc)
                print(f"epoch{t // iter_per_epoch:.0f} train_acc： {train_acc:.3f}  test_acc： {test_acc:.3f}")

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

