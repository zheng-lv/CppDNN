#include <iostream>
#include <eigen3/Eigen/Dense>
#include <functional>
#include <vector>
//#include "DeepNueralNetwork.h"
using Eigen::MatrixXd;
using std::cout;
using std::endl;
using std::vector;

#include <CppDNN/DeepNueralNetwork.h>

int main()
{
    // 定义一个 DeepNueralNetwork 类型的对象 dnn，表示神经网络
    DeepNueralNetwork dnn;
    // 从指定路径读取使用 Keras 训练的神经网络参数
    dnn.ReadFromKeras("/home/nader/workspace/github/CppDNN/example/keras_simple/simple.txt");
    // 创建一个 5x1 的矩阵 input，用于输入数据
    MatrixXd input(5,1);
    input(0,0) = 1;
    input(1,0) = 2;
    input(2,0) = 1;
    input(3,0) = 2;
    input(4,0) = 1;
    // 调用 Calculate 方法进行神经网络的计算
    dnn.Calculate(input);
    // 输出神经网络的计算结果
    cout<<dnn.mOutput<<std::endl;
}
