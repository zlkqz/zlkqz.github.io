---
title: Megatron-LM
math: true
date: 2023-08-02
---



- Megatron-LM的张量并行（TP）思想是：**将一个权重Tensor沿行或者列进行划分，将各部分分别在不同的卡上进行计算，最后再汇总**
- 相比于其他张量并行的方法，Megatron-LM的优点主要是：**不需要改重写底层算子，只需要稍微改变一点计算和增加少量的同步锁即可。并且可以和DP、PP等方法一起使用**



# 1 划分方法

- 一个权重张量可以延行或者列展开，针对Transformer里的MLP层和Attention层具有不同的划分方式



### 1.1 MLP层

- 计算流程如下：

<img src="https://zlkqzimg-1310374208.cos.ap-chengdu.myqcloud.com/image-20230803125202209.png" alt="image-20230803125202209" style="zoom:50%;" />

- 其中输入的X和输出的Z是每张卡都有一份copy的
- 权重矩阵A按列划分，B是按行划分，算出$$Z_1, Z_2$$后在$$g$$进行一个**All-Reduce**，每张卡得到相同的$$Z = Z_1 + Z_2$$，再每张卡都对$$Z$$重复一次相同的dropout
- 其中$$f$$在forward时没用，在backward时代表对梯度做一次**All-Reduce**



### 1.2 Attention层

- 计算流程如下：

<img src="https://zlkqzimg-1310374208.cos.ap-chengdu.myqcloud.com/image-20230803130712170.png" alt="image-20230803130712170" style="zoom: 28%;" />

- 注意：**这里对Q、K、V的划分是按Attention Head划分的，每张卡会包含1到多个Head**

- 当然，应该尽量保证GPU数能整除Attention Head的数量，尽量均匀划分



### 1.3 通信量分析

- 两种子层的划分，**通信量在于forward时$$g$$的All-Reduce和backward时$$f$$的All-Reduce**

- 一次All-Reduce的通信量为：**2 * batch_size * seq_len * hidden_size**

- 所以一个MLP层或Attention层的通信量都为：**4 * batch_size * seq_len * hidden_size**

- 回想一下ZeRO中，通信量为2倍模型参数（ZeRo Stage2，通信的就只有梯度），**所以两者的通信量是差不多的**



### 1.4 Embedding层

- 在NLP模型参数中，由于词表一般都很大，所以很大一部分参数都集中在Embedding矩阵中，所以对Embedding矩阵进行划分也是十分必要的
- Embedding矩阵同时用于最开始的word2vec和最后的输出层（Weight Tying），所以两个层对Embedding矩阵划分后的操作是不同的



##### 1.4.1 输入层

- Embedding矩阵shape为（hidden_size, vocab_size），将其按列划分
- 将完整的词索引向量输入进每张卡，对每张卡的Embedding矩阵进行提取，**若在某张卡上没有找到对应的Embedding（因为该词的Embedding在其他卡上），则直接将对应的提取值置为全0，最后一起拿去All-Reduce即可**

<img src="https://zlkqzimg-1310374208.cos.ap-chengdu.myqcloud.com/image-20230803134548873.png" alt="image-20230803134548873" style="zoom:33%;" />

- 我们一般只对word embedding进行划分，position embedding的矩阵之类的由于并不大，所以每张卡都备份一份完整的



##### 1.4.2 输出层

- 同样的，将Embedding矩阵按列划分，假设只有两张卡，$$E = [E_1, E_2]$$
- 然后计算$$Y_1, Y_2 = XE_1, XE_2$$，每张卡得到一半的logits，然后进行**All-Gather**操作，每张卡得到完整的logits，再用交叉熵算损失
- 但是这样的方法通信量为**batch_size * seq_len * vocab_size**，通行量过大
- **改进方法：**

> 1. 得到$$Y_1, Y_2 = XE_1, XE_2$$的部分logits之后，每张卡对自己所属的部分logits算$$\sum_{logit}e^{logit}$$，然后将这个值**All-Reduce**
> 2. 那么现在每张卡都得到了完整的$$\sum_{logit}e^{logit}$$，就可以算自己这块对应的loss了，然后再把这个loss再**All-Reduce**一次，每张卡即可得到完整的loss
> 3. 改进之后通信量约为**batch_size * seq_len**





# 2 2D-Parallelism（DP+TP）

- 前面说过TP和DP的通信量基本差不多，而在多机多卡的场景下，**我们对体系设计主要考虑通信量和是否需要等待其他机器两个因素**
- 2D-Parallelism体系如下：

<img src="https://zlkqzimg-1310374208.cos.ap-chengdu.myqcloud.com/image-20230803143148059.png" alt="image-20230803143148059" style="zoom:40%;" />

- 即一台机器输入一批micro-batch，然后一台机器上的多张卡来做模型并行
- 这样做的好处是：**对于TP，每次每张卡分别算了一部分参数的梯度后，需要做All-Reduce操作才能继续上一层的backward，所以对于带宽要求较高，理应放在同一台机器中；而对于DP，算完属于自己micro-batch的部分梯度后，直接把这部分梯度传出去就可以了，可以同时进行上一层的backward，所以不需要等待，对带宽要求不高**

- 值得一提的是，同样有3D-Parallelism，体系如下，同样是在多机之间做DP：

<img src="https://zlkqzimg-1310374208.cos.ap-chengdu.myqcloud.com/image-20230804180339179.png" alt="image-20230804180339179" style="zoom:50%;" />



# 3 实验效果

- 要评测采用TP或者DP+TP是否降低了训练效率，最直观的方法就是看每张卡的吞吐量：

<img src="https://zlkqzimg-1310374208.cos.ap-chengdu.myqcloud.com/image-20230803143930235.png" alt="image-20230803143930235" style="zoom:50%;" />

图中Y轴为吞吐量相比于一张卡训练的时候的吞吐量的占比