import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()#兼容1.0版本
import numpy as np
import random
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# 属性设置
trainNum = 55000
testNum = 10000
trainSize =500
testSize = 5
k = 4
# data 分解  1、范围0~trainNum； 2、trainSize； 3、replace=False
trainIndex = np.random.choice(trainNum, trainSize, replace=False)
testIndex = np.random.choice(testNum, testSize, replace=False)
trainData = mnist.train.images[trainIndex] #训练图片；trainData= (500, 784) 500是图片个数，图片宽28*高28=784
trainlabel = mnist.train.labels[trainIndex] #训练标签；trainlabel= (500, 10)
testData = mnist.test.images[testIndex]# testData= (5, 784)
testLabel = mnist.test.labels[testIndex]# testLabel= (5, 10)
print ("trainData=",trainData.shape)
print ("trainlabel=",trainlabel.shape)
print ("testData=",testData.shape)
print ("testLabel=",testLabel.shape)

# tf input
trainDataInput = tf.placeholder(shape=[None, 784], dtype=tf.float32)# shape为维度
trainLabelInput = tf.placeholder(shape=[None, 10], dtype=tf.float32)
testDataInput = tf.placeholder(shape=[None, 784], dtype=tf.float32)# shape为维度
testLabelInput = tf.placeholder(shape=[None, 10], dtype=tf.float32)

# knn distance 原5*785————>现5*1*784
# 5测试数据， 500训练数据， 每个维度都是784（3D） 2500*784
f1 = tf.expand_dims(testDataInput, 1) #夸大一个维度
f2 = tf.subtract(trainDataInput, f1) #784 sum(784)
f3 = tf.reduce_sum(tf.abs(f2), reduction_indices=2) #完成数据累加 784
f4 = tf.negative(f3) # 取反
f5, f6 = tf.nn.top_k(f4, k=4) # 选取f4 最大的四个值
f7 = tf.gather(trainLabelInput, f6) # 根据下标所引训练图片的标签
f8 = tf.reduce_sum(f7, reduction_indices=1)
f9 = tf.argmax(f8, dimension=1) # tf.argmax 选取在某一个最大的值

with tf.Session() as sess:
    p1 = sess.run(f1, feed_dict={testDataInput:testData[0:5]})
    print ("p1 = ",p1.shape) # p1 =  (5, 1, 784)
    p2 = sess.run(f2, feed_dict={trainDataInput:trainData, testDataInput:testData[0:5]})
    print ("p2 = ",p2.shape) # p2 =  (5, 500, 784)
    p3 = sess.run(f3, feed_dict={trainDataInput:trainData, testDataInput:testData[0:5]})
    print ("p3 = ",p3.shape) # p3 =  (5, 500)
    print ("p3[0, 0] = ", p3[0, 0]) # p3[0, 0] =  116.76471
    p4 = sess.run(f4, feed_dict={trainDataInput:trainData, testDataInput:testData[0:5]})
    print ("p4 = ", p4.shape) # p4 =  (5, 500)
    print ("p4[0, 0] = ", p4[0, 0]) # p4[0, 0] =  -116.76471
    p5, p6 = sess.run((f5, f6), feed_dict={trainDataInput:trainData, testDataInput:testData[0:5]})
    print ("p5 = ",p5.shape) # p5 =  (5, 4)  每一张测试图片（5张） 分别对应4张最近训练图片
    print ("p6 = ",p6.shape) # p6 =  (5, 4)
    print ("p5[0, 0] = ", p5[0, 0]) # 这是一个随机数
    print ("p6[0, 0] = ", p6[0, 0]) # p6 index
    p7 = sess.run(f7, feed_dict={trainDataInput:trainData, testDataInput:testData[0:5], trainLabelInput:trainlabel})
    print ("p7 = ", p7.shape) # p7 =  (5, 4, 10)
    p8 = sess.run(f8, feed_dict={trainDataInput:trainData, testDataInput:testData[0:5], trainLabelInput:trainlabel})
    print ("p8 = ", p8)
    print ("p8.shape = ", p8.shape) # p8.shape =  (5, 10)
    p9 = sess.run(f9, feed_dict={trainDataInput:trainData, testDataInput:testData[0:5], trainLabelInput:trainlabel})
    print ("p9 = ", p9) # p9 =  [3 3 2 8 2]， 是p8中最大值的下标
    print ("p9.shape = ", p9.shape) # p9.shape =  (5,)
    p10 = np.argmax(testLabel[0:5], axis=1) # 测试标签的索引内容
    print ("p10 = ", p10) # 通过比较p9和p10的结果得到统计的概率

#计算统计的识别正确率
j = 0
for i in range(0, 5):
    if p10[i] == p9[i]:
        j = j + 1
print ("本次识别正确率 =", j*100/5)

