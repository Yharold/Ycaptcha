# 项目分析

这是一个验证码识别项目，原本的项目仓库是![GiantPandaCV验证码识别竞赛解决方案](https://github.com/pprp/captcha.Pytorch)。主要的目标就是创建一个模型，训练模型，然后测试模型的准确率。需要实现的功能有：
1. 数据增强：需要对原始的数据进行数据增强，增多模型的训练数据。
2. 模型创建：需要创建多个模型，分别验证不同模型的效果。
3. 损失函数：对多种损失函数进行验证
4. 优化器：对多个优化器进行验证
5. 学习率：对学习率进行验证
6. 训练和测试：训练模型，保存模型，并测试模型的准确率。 

## 数据增强
数据增强的方式有很多种，旋转，裁剪，缩放，翻转，噪声等。原始的数据增强方式有5种，分别是：
- 扭曲
- 数据9:1划分+扩增3倍
- 扭曲+缩放
- 倾斜+扩增两倍 
- 扭曲+缩放+倾斜+扩增两倍
- 9:1划分+倾斜 
原始数据训练集和测试集是8:2划分，所以这里的9:1同样是对数据的扩增。 

我这里只用一种方式，就是扭曲+缩放。文件是utils/dataAug.py。这个文件中实现了两种数据增强方式，扭曲+缩放和扭曲+缩放+倾斜。

## 模型创建
原始的模型只使用ResNet18和ResNet34，但后面作者说会测试attention，ibn，bnneck等。作者这里的Resnet18模型有一些修改，第一个卷积变成了3x3的大小，去掉了最大池化层，添加了dropout层，最后通过4个全连接层输出4个字符。

我这里的模型是使用了作者的ResNet18和原生的ResNet18，主要目标是做一个对比。

## 损失函数
损失函数就用交叉熵损失函数(CrossEntropyLoss)即可。

## 优化器
作者使用的优化器有三个，Adam，RAdam和AdamW。
我这里使用RAdam和AdamW。目的依旧是为了做一个对比

## 学习率
作者这里的学习率有多个，具体可以看最后的那张表。我这里使用的学习率是0.001，调度器的类是在lib/scheduler.py。它使用warmup和weight decay。

## 训练和测试

训练和测试需要注意的就是如何保存记录。这里我就使用TensorBoard记录训练过程。另外就是使用GPU加速训练。

# 项目结构
项目结构如下：
- root:
  - datasets:存放数据
    - train：训练数据
    - test：测试数据
    - auged_train_0：第一种数据增强方式
    - auged_train_1：第二种数据增强方式
  - logs：存放TensorBoard记录
  - models：存放模型类
    - model.py: 模型类
  - weights：存放模型权重
  - config: 存放配置文件
    - parameter.py：参数配置,包括数据集路径，训练轮数，批量大小，学习率，模型保存路径等。
  - utils: 存放工具类
    -  dataAug.py: 数据增强类
  - lib: 存放损失函数，优化器，调度器，读取数据等
    - loss.py：损失函数
    - optimizer.py：优化器
    - scheduler.py：学习率调度器
    - dataset.py：读取数据 
  - train.py：训练脚本
  - test.py：测试脚本
  - predict.py：预测脚本
  - main.py：主函数

# 实验计划
训练分为下面几种情况：
1 RawResNet+lr=0.001+epoch=200+batch_size=64+RAdam+auged0=64 
2 ResNet+lr=0.001+epoch=200+batch_size=64+RAdam+auged0=64 
3 ResNet+lr=0.001+epoch=200+batch_size=64+RAdam+auged1=64
4 ResNet+lr=0.001+epoch=200+batch_size=64+AdamW+auged0=64 
5 ResNet+lr=0.0035+epoch=200+batch_size=64+RAdam+auged0=64 
6 ResNet+lr=0.0005+epoch=200+batch_size=64+RAdam+auged0=64 

1 2对比选出最优的模型，23对比选出最好的数据增强方式，24对比选出最好的优化器，2,5,6对比选择好的学习率。
实际是做完一个对比才能有下一个对比。

# RAdam

RAdam是一种改进的Adam优化器，论文地址是![RAdam论文地址]https://arxiv.org/pdf/1908.03265.pdf
具体的算法如下：
![RAdam算法](fig/Radam.png)

下面是RAdam的实现代码