### 1. 数据处理模块    
（1）加载数据（查看数据shape和type，查看数据的一个example）    
（2）训练集和测试集特征缩放（归一化或标准化，需要的话做reshape操作）    
（3）数据标签格式化（把数据label进行one_hot编码，或者处理"rank 1 array"）    
（4）制作mini-batch数据集   
   
  
### 2. 构造计算图：
#### 2.1 Tensorflow1.x版本默认构建静态图(static graph)      
+ 使用w = tf.Variable()定义模型参数
+ 使用x = tf.placeholder()定义模型的输入(未知的常量)
+ 使用x = tf.constant()构造已知的常量
+ 构造计算图+设计损失函数
+ 用train = tf.train构造优化模型（损失函数+优化算法+学习率）  
  
+ 用init = tf.global_variables_initializer()构建模型参数初始化器
+ 用with tf.Session() as sess:构建会话
+ 用sess.run(init)来给参数赋初始值
+ 用sess.run(train, feed_dict = {x: input})使得模型开始训练

#### 2.2 Tensorflow2.x版本默认Eager Excution，构造动态图
+ keras.Sequential构建模型  
+ compile(配置优化器、损失函数和评估函数)  
+ fit训练  
+ evaluate评估  
+ predict预测  

#### 2.3 Tensorflow2.x版本构造静态图AutoGraph
使用@tf.funciton自动构造静态图（AutoGraph）：    
+ （1）Sequential构建模型  
+ （2）配置优化器、损失函数和评估函数  
+ （3）使用tf.GradientTape进行训练  
初始化权重参数，设置学习率；     
前向和反向传播函数：前向传播计算损失函数-->反向传播求各参数的偏导数    
优化过程：设置训练for循环-->更新权重参数-->隔一定步长时输出信息    
模型训练结束，在测试集上做预测，评估模型性能    
辅助训练方法：绘制训练集损失函数，测试集损失函数的变化曲线   



 
#### 参考资料：
1. https://www.tensorflow.org/tutorials/
2. https://www.tensorflow.org/guide/

