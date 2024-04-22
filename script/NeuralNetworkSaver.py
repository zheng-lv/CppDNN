def NeuralNetworkSaver(ns, layers: list, save_in: str):
    """
    函数用于将神经网络的信息保存到指定的文件中

    参数：
    ns -- 包含神经网络参数的列表
    layers -- 包含每一层信息的列表
    save_in -- 保存文件的路径
    """
    file = open(save_in, "w")  # 以写入模式打开文件
    file.write("# Layer Numbers: " + str(len(layers)))  # 写入层的数量
    file.write('\n')  # 换行

    for i, layer in enumerate(layers):  # 遍历每一层
        file.write("# Layer Number: " + str(i) + "\n")  # 写入层号
        file.write(layer[0] + "\n")  # 写入层的类型

        file.write(str(ns[i + 1]) + " " + str(ns[i]) + '\n')  # 写入权重和上一层节点数的信息

        file.write("# W\n")  # 写入权重部分的标记
        file.write(tensor_to_str(layer[1]))  # 将权重转换为字符串并写入

        file.write("# B\n")  # 写入偏置部分的标记
        file.write(tensor_to_str(layer[2]))  # 将偏置转换为字符串并写入


def tensor_to_str(tensor):
    """
    函数用于将张量转换为字符串形式

    参数：
    tensor -- 要转换的张量

    返回：
    转换后的字符串
    """
    out = ""  # 用于存储转换后的字符串
    for i in range(len(tensor)):  # 遍历张量的每一行
        for j in range(len(tensor[i])):  # 遍历每行的每个元素
            out += str(tensor[i][j]) + "\n"  # 将元素转换为字符串并添加到结果中
    return out  # 返回转换后的字符串
