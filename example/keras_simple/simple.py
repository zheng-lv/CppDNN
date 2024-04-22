# 从 keras 库中导入 models 和 layers 模块
from keras import models
from keras import layers
# 从 numpy 库中导入 array 模块
from numpy import array

mnistfile = open('mnist_example', 'r')  # 打开名为 "mnist_example" 的文件，并以读取模式打开
mnistdata = mnistfile.read()  # 读取文件内容
mnistdata = mnistdata.splitlines()[0].split(' ')  # 将读取的内容按行分割，并取第一行，再按空格分割
mnistdataf = []  # 创建一个空列表用于存储处理后的数据
for m in mnistdata:  # 遍历分割后的每个元素
    mnistdataf.append(float(m))  # 将每个元素转换为浮点数并添加到列表中
mnistdata = array(mnistdataf)  # 将列表转换为数组
mnistdata = mnistdata.reshape((1, 784))  # 将数组形状调整为 (1, 784)

mnistdata = [1, 2, 1, 2, 1]  # 定义新的 mnistdata 数据
mnistdata = array(mnistdata)  # 将数据转换为数组
mnistdata = mnistdata.reshape((1, 5))  # 调整数组形状为 (1, 5)

# 创建一个顺序模型
network = models.Sequential() 
# 在模型中添加一个具有 8 个神经元的密集连接层，输入形状为 (5,)
network.add(layers.Dense(8, input_shape=(5,))) 
# 添加一个具有 5 个神经元的密集连接层，激活函数为 relu
network.add(layers.Dense(5, activation='relu')) 
# 添加一个具有 2 个神经元的密集连接层，激活函数为 softmax
network.add(layers.Dense(2, activation='softmax')) 
# 编译模型，设置优化器为 rmsprop，损失函数为 categorical_crossentropy，评估指标为 accuracy
network.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy']) 

print(network.predict(mnistdata))  # 打印模型对 mnistdata 的预测结果

# 保存模型到 simple.h5 文件
network.save('simple.h5')  # 
