import sys
from keras import models# 导入 Keras 中的 models 模块

#首先检查命令行参数的数量是否足够，如果不够则打印用法信息并退出程序。
if len(sys.argv) < 3:# 检查命令行参数的数量是否小于 3
    print('usage: python DecodeKerasModel.py input output')# 如果数量不够，打印用法信息
    exit(1)
#读取输入和输出的文件名。
input = sys.argv[1]# 获取第一个命令行参数，即输入文件的名称
output = sys.argv[2]# 获取第二个命令行参数，即输出文件的名称
print(input)# 打印输入文件的名称
#打开输出文件用于写入数据。
outputFile = open(output, 'w')# 以写入模式打开输出文件
#加载输入的 Keras 模型。
model = models.load_model(input)# 加载输入的 Keras 模型
weights_list = model.get_weights()# 获取模型的权重列表
print("#################################################################")
print("# Layer Numbers: " + str(len(weights_list)) + '\n')# 打印层的数量
outputFile.write("# Layer Numbers: " + str(int(len(weights_list)/2)) + '\n') # 将层的数量写入输出文件
for l in range(int(len(weights_list)/2)):#写入层的数量。
    w = weights_list[l * 2]# 获取当前层的权重矩阵
    b = weights_list[l * 2 + 1]# 获取当前层的偏置向量
    outputFile.write("# Layer Number: {}".format(l) + '\n')# 在输出文件中写入当前层的编号
    print("# Layer Number: {}".format(l) + '\n')# 在控制台打印当前层的编号
    outputFile.write(model.layers[l].activation.__str__().split(' ')[1] + '\n')# 在输出文件中写入当前层的激活函数
    outputFile.write(str(len(b)) + ' ' + str(len(w)) + '\n')# 在输出文件中写入偏置向量和权重矩阵的大小
    print(str(len(b)) + ' ' + str(len(w)) + '\n')# 在控制台打印偏置向量和权重矩阵的大小
    outputFile.write("# W" + '\n')# 在输出文件中写入 "W" 表示开始写入权重矩阵
    print(w.shape)# 在控制台打印权重矩阵的形状
    for x in w:# 遍历权重矩阵的每个元素
        for y in x:# 遍历矩阵中的每个元素
            outputFile.write(str(y) + '\n')# 将元素写入输出文件
    outputFile.write("# B" + '\n')# 在输出文件中写入 "B" 表示开始写入偏置向量
    print(b.shape)# 在控制台打印偏置向量的形状
    for x in b:# 遍历偏置向量的每个元素
        outputFile.write(str(x) + '\n')# 将元素写入输出文件
        
outputFile.close()
# 关闭输出文件
