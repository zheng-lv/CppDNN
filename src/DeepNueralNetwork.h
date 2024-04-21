#pragma once


#include "Layer.h"
#include <vector>
#include <fstream>
#include <sstream>
#include <string>

class DeepNueralNetwork{
public:
    MatrixXd mInput;
    MatrixXd mOutput;
    std::vector<Layer> mLayers;

// 添加新层到网络
    void AddLayer(const Layer layer)
    {
        mLayers.push_back(layer);
    }
// 执行神经网络前向传播计算
    void Calculate()
    {
        MatrixXd * out = &mInput;// 开始时使用输入数据
        for(size_t l = 0; l < mLayers.size(); l++)
        {
//             std::cout<<"calc l"<<std::endl;
             mLayers[l].Calculate(*out);// 计算当前层的输出
             out = &(mLayers[l].mOutput);// 将当前层输出设置为下一层的输入
//             std::cout<<"out"<<(*out)(0,0)<<","<<(*out)(1,0)<<std::endl;
        }
        mOutput = (*out);// 最终输出
    }

    void Calculate(MatrixXd input)
    {
        mInput = input;
        Calculate();
    }

// 从 Keras 模型文件读取网络结构和权重
    bool ReadFromKeras(std::string file)
    {
        std::fstream infile(file);// 打开文件
        std::string line;
        std::getline(infile, line);// 读取层数
        int layerSize = std::stod(line.substr(line.find_last_of(" "), line.size()));// 转换层数为整数
        std::cout << "layerSize: " << layerSize << std::endl;
        for (int l = 0; l < layerSize; l++)
        {// 读取并解析每层的数据
            std::getline(infile, line);// 激活函数
            std::string activation;
            std::getline(infile, line);
            std::istringstream acc(line);
            acc >> activation;// 获取激活函数名称
		
            std::getline(infile, line);// 权重尺寸
            std::istringstream iss(line);
            int mSize, nSize;
            iss >> mSize >> nSize;// 权重矩阵的行和列
            std::getline(infile, line);
            MatrixXd W(mSize, nSize);// 创建权重矩阵
            for (int n = 0; n < nSize; n++)
            {
                for (int m = 0; m < mSize; m++)
                {
                    std::getline(infile, line);
                    std::istringstream iss(line);
                    double w; iss >> w;// 读取权重值
                    W(m, n) = w;// 设置权重矩阵
                }
            }
            std::getline(infile, line);// 读取偏置
            MatrixXd B(mSize, 1);// 创建偏置矩阵
            for (int m = 0; m < mSize; m++)
            {
                std::getline(infile, line);
                std::istringstream iss(line);
                double b; iss >> b;// 读取偏置值
                B(m, 0) = b;// 设置偏置矩阵
            }
            AddLayer(Layer(W, B, StringToFunction(activation)));// 添加层到网络
        }
        return true; //for remove warning
    }

//目前，该方法只是简单地调用了 ReadFromKeras 方法，这表明开发者可能假设TensorFlow模型文件的格式与Keras模型文件相同，或者是为了测试而暂时这样处理。
    bool ReadFromTensorFlow(std::string file)
    {
		ReadFromKeras(file);// 调用ReadFromKeras方法处理文件
		return true; //for remove warning
    }
};

