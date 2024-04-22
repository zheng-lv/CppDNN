import tensorflow as tf
import NeuralNetworkSaver as nns  # 导入 NeuralNetworkSaver 模块

# 输入节点数
nx = 94 
# 第一层隐藏层节点数
n1 = 256 
# 第二层隐藏层节点数
n2 = 64 
# 原本可能有第三层隐藏层节点数，这里注释掉了
# n3 = 32 
# 输出节点数
n4 = 11 

with tf.variable_scope("Layer1"): 
    w1 = tf.Variable(tf.random_normal([nx, n1]), name="weight_1")  # 生成第一层权重的随机值
    b1 = tf.Variable(tf.random_normal([1, n1]), name="bias_1")  # 生成第一层偏置的随机值
    # o1 = tf.nn.relu(tf.add(tf.matmul(x, w1), b1, name="o1"))  # 应用激活函数

with tf.variable_scope("Layer2"): 
    w2 = tf.Variable(tf.random_normal([n1, n2]), name="weight_2")  # 生成第二层权重的随机值
    b2 = tf.Variable(tf.random_normal([1, n2]), name="bias_2")  # 生成第二层偏置的随机值
    # o2 = tf.nn.relu(tf.add(tf.matmul(o1, w2), b2, name="o2"))  # 应用激活函数

with tf.variable_scope("Out"): 
    w4 = tf.Variable(tf.random_normal([n2, n4]), name="weight_out")  # 生成输出层权重的随机值
    b4 = tf.Variable(tf.random_normal([1, n4]), name="bias_out")  # 生成输出层偏置的随机值
    # o4 = (tf.add(tf.matmul(o2, w4), b4, name="o4"))  # 生成输出

saver = tf.train.Saver()  # 创建保存器
init = tf.global_variables_initializer()  # 初始化所有变量

with tf.Session() as sess: 
    sess.run(init)  # 运行初始化操作
    saver.restore(sess, './logs/SaveforLoad/SL.ckpt')  # 从指定路径加载保存的模型参数

    _w1, _w2, _w4, _b1, _b2, _b4 = sess.run([w1, w2, w4, b1, b2, b4])  # 运行会话以获取权重和偏置值
    nns.NeuralNetworkSaver([nx, n1, n2, n4], [["relu", _w1, _b1], ["relu", _w2, _b2], ["linear", _w4, _b4]], "test")  # 使用 NeuralNetworkSaver 函数保存网络参数
