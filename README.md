ado-tensorflow-models

用的tensorflow中高级API如：estimator、layers、dataset等

在coding中对于自己发现的不清楚的希望一直能增加详细的注释说明。

由于个人经验得欠缺有不对的，还希望留言指出，谢谢。

针对绝大部分的模型或者框架，会在CSDN上做一个简要的说明，也可移步查阅。[传送门CSDN](https://blog.csdn.net/WUUUSHAO)
## Requirements
Python 3

Tensorflow >= 1.8


# 已完成
## [TextCNN](https://github.com/adowu/ado-tensorflow-models/tree/master/01_TextCNN)  
相关论文地址：  
[Convolutional Neural Networks for Sentence Classification](https://arxiv.org/abs/1408.5882)  
[Highway Networks](https://arxiv.org/abs/1505.00387)  
应用的场景是餐饮评价类语料的三分类任务

##  [TextRNN](https://github.com/adowu/ado-tensorflow-models/tree/master/02_TextRNN) 
相关论文地址：  
[Recurrent Neural Network for Text Classification with Multi-Task Learning](https://arxiv.org/abs/1605.05101v1)     
应用的场景是餐饮评价类语料的三分类任务

## [AllRNN](https://github.com/adowu/ado-tensorflow-models/tree/master/03_AllRNN)
01. [basic_rnn](https://github.com/adowu/ado-tensorflow-models/blob/master/03_AllRNN/basic_rnn_demo.py)  
02. [basic_lstm](https://github.com/adowu/ado-tensorflow-models/blob/master/03_AllRNN/basic_lstm_demo.py)  
03. [multi_rnn](https://github.com/adowu/ado-tensorflow-models/blob/master/03_AllRNN/multi_rnn_demo.py)，相关的论文[RECURRENT NEURAL NETWORK REGULARIZATION](https://arxiv.org/pdf/1409.2329.pdf)

##  [Attention](https://github.com/adowu/ado-tensorflow-models/tree/master/04_AttentionRNN)   
01. [BahdanauAttention](https://arxiv.org/abs/1409.0473) 
02. [LuongAttention](https://arxiv.org/abs/1508.04025)

##  [LinearRegression](https://github.com/adowu/ado-tensorflow-models/tree/master/08_LinearRegression)

