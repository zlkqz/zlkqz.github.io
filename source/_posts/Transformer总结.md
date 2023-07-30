---
title: Transformer总结
math: true
date: 2022-1-25
---

- Transformer摒弃了传统的CNN/RNN模型，而是用纯注意力机制， **相对于RNN，实现了并行化，并且消除了memory对于距离的依赖性（无法捕捉长距离依赖）。**





# 1 注意力机制

- **注意力机制中分别有key、query、value（一般key=value），通过key、query之间的相似度，计算得到每个value对应的权值，再对所有value加权求和，得到一整个序列的表征。其中对于自己本身的注意力机制称为self-attention（自注意力机制），即key=value=query**



### 1.1 Scaled Dot-Product Attention(点积)

<img src="https://zlkqzimg-1310374208.cos.ap-chengdu.myqcloud.com/image-20220512112222285.png" alt="image-20220512112222285" style="zoom: 37%;" />

- 在计算时，我们是将query、key、value（分别为$$d_k, d_k, d_v$$维）打包成Q，K$$\in R^{N \times d_k}$$，V$$\in R^{N \times d_v}$$，具体做法是：

> 将送进来的输入input$$\in R^{N \times d_{model} }$$（其中$$d_{model}$$为embedding的维度，且q、k、v三者的input可能各自不同），input分别乘$$W^Q、W^K \in R^{d_{model} \times d_k}$$，$$W^V \in R^{ {d_{model} \times d_v} }$$即可得到Q、K、V

- 在计算权值时，将Q、K相乘，再除以$$\sqrt{d_k}$$，再softmax得到权值。**除以$$\sqrt{d_k}$$的原因**：

> **维度过大会使Q、K相乘的结果过大，容易把softmax的区域推向梯度极小的区域。并且实验证明在$$d_k$$较小时，其实除不除效果差不多**

- 得到权重后再和V相乘，总过程为：

$$
\operatorname{Attention}(Q, K, V)=\operatorname{softmax}\left(\frac{Q K^{T} }{\sqrt{d_{k} }}\right) V
$$

- 还有一种较常用的注意力机制叫Additive attention， 是使用一个单隐藏层的全连接网络计算权重，两者效果差不多，**但是dot-product会快得多**



### 1.2 Multi-Head Attention(多头注意力机制)

<img src="https://zlkqzimg-1310374208.cos.ap-chengdu.myqcloud.com/image-20220512112235945.png" alt="image-20220512112235945" style="zoom:35%;" />

- 多头注意力机制本质上就是**做多次Scaled Dot-Product Attention**
- 具体做法是：

> **重复做h次Scaled Dot-Product Attention（每次的W权重矩阵分别独立），将每次得到的结果$$Z \in R^{N \times d_v}$$在第二维连结，形状变为$$R^{N \times hd_v}$$，再乘一个$$W^O \in R^{hd_v \times d_{model} }$$，即可得到形状为$$R^{N \times d_{model} }$$的最终结果**

- 总过程为：

$$
\begin{aligned}
\operatorname{MultiHead}(Q, K, V) &=\operatorname{Concat}\left(\text { head }_{1}, \ldots, \text { head }_{\mathrm{h} }\right) W^{O} \\
\text { where head } &=\operatorname{Attention}\left(Q W_{i}^{Q}, K W_{i}^{K}, V W_{i}^{V}\right)
\end{aligned}
$$

- 在base模型中取的$$h = 8$$，且$$d_k = d_v = d_{model}/h = 64$$

- **多头注意力的好处：类似于CNN中的通道，能提取到不同子空间下的特征。多头注意力允许模型共同关注来自不同位置的不同表示子空间的信息。**如果单纯使用单注意力头+平均化，会抑制这一点

> Multi-head attention allows the model to jointly attend to information from different representation subspaces at different positions. With a single attention head, averaging inhibits this.  



### 1.3 自注意力机制的好处

- 自注意力机制最大的好处肯定是实现了**并行化，加快了训练速度**。并且得到的结果相比于其他方法（如全局平均池化），**更具有解释性**，self-attention是可以退化成平均的，所以结果肯定好于平均。

- 论文从每层的总计算复杂度、可并行化的计算数量（用顺序操作的最小量来衡量）、长距离依赖的距离三个方面进行了对比：

![image-20220512114401561](https://zlkqzimg-1310374208.cos.ap-chengdu.myqcloud.com/image-20220512114401561.png)

- **并且单个注意力头不仅清楚地学习执行不同的任务，而且许多似乎表现出与句子的句法和语义结构相关的行为**

> Not only do individual attention heads clearly learn to perform different tasks, many appear to exhibit behavior related to the syntactic and semantic structure of the sentences  





# 2 模型结构

<img src="https://zlkqzimg-1310374208.cos.ap-chengdu.myqcloud.com/image-20220512121448150.png" alt="image-20220512121448150" style="zoom:28%;" />

- 图中多头注意力层的输入从左到右依次为V、K、Q

### 2.1 Encoder和Decoder

- Encoder由N=6个相同的块组成，每个块有两个子层，一个多头注意力层，一个全连接层，子层输入输出都是都是$$R^{N \times d_{model} }$$，其中N为时间步个数，也就是词数。**并且在每个子层都有一个残差结构+LayerNorm**，先残差后LayerNorm：

$$
LayerNorm(x + Sublayer(x))
$$

- Decoder同样是由N=6个相同的块组成，每个块有3个子层，有两个和Encoder中一模一样。增加了一个带Mask的多头注意力层。**Decoder最开始的输入在训练时和预测时不一样**，在**训练时是把所有的翻译结果一次性输入**，并行化提高速度。而**预测时是类似于RNN一样的串行方式**，第一次给Decoder输入句子的开始符号，然后得到第一个翻译结果，再将第一个翻译结果当作输入送入Decoder。总结来说就是：**每次Decoder的输入为之前所有时间步的结果**。而在训练时，是一次导入所有结果，所以需要**Mask掉未来时间步的翻译结果**。



### 2.2 多头注意力层

- 进行的操作其实就是上文提到的多头注意力机制：**将输入分别乘一个矩阵W，转换成Q、K、V，再计算权重并加权平均，得到Z。将上述过程进行h次，每次使用的是相互独立的W，再将Z连结，再乘一个权重矩阵，得到最终结果。**

- 需要注意的是Decoder中的Masked Multi-Head Attention。我们在**预测时，肯定是无法知道未来的信息的（也就是之后时间步的输出），但是在训练时我们是将翻译结果一次性使用一个矩阵导入的**。所以为了保持一致性，我们需要在**训练时屏蔽掉未来的信息，即当前时间步t的输出只取决于t-1及其之前的时间步。**
- 下方为一个Attention Map，每个单元代表该行对四个列对应的权值。如第一行代表"I"分别对"I"、"have"、"a"、"dream"的权值。

<img src="https://zlkqzimg-1310374208.cos.ap-chengdu.myqcloud.com/64.jpg" alt="64" style="zoom:50%;" />

显然在通过"I"预测"have"时，是不知道后面的"have"、"a"、"dream"的，所以需要通过Mask屏蔽掉未来的信息，其他时间步的时候类似：

<img src="https://zlkqzimg-1310374208.cos.ap-chengdu.myqcloud.com/56.jpg" alt="56" style="zoom:50%;" />

上图就是经过Mask后的Attention Map，将每个时间步未来的信息进行了屏蔽，具体的做法是：**在计算V的权重时，softmax之前将对应的值设为$$-\infty$$**



### 2.3 全连接层

- 每个全连接子层有两个层，进行的运算为：

$$
\operatorname{FFN}(x)=\max \left(0, x W_{1}+b_{1}\right) W_{2}+b_{2}
$$

第二层是没有激活函数的

- 层的输入为$$d_{model} = 512$$，经过第一层变为$$d_{ff} = 2048$$，经过第二层又变为512维。



### 2.4 Layer Normalization

- BN是对一个batch-size样本内的每个特征做归一化，LN是对每个样本的每个时间步中所有特征做归一化

- **使用LN，假设输入X，X.shape = (batch_size,time_step,embedding_dim) 那么mean.shape = std.shape = (batch_size,time_step,1)。即对embedding维做归一化，另外LN中同样是有放缩的参数的**
- 而使用BN，那么mean.shape = std.shape = (1,time_step,embedding_dim)，即对batch维做归一化

- **选用LN而弃用BN的原因：**BN需要较大的batch_size来保证对于期望、方差的统计的可靠性，对于CNN、Dense层来说则好统计。但是在天然变长的NLP任务中，如果选用BN需要**对每个时间步的状态进行统计**，这会导致在偏长的序列中，**靠后的时间步的统计量不足**。相比之下使用LN则不会有这种限制

- **而对embedding层进行归一化也更具有解释性，因为embedding层的每个值都是直接相关的**



### 2.5 词嵌入

- Transformer中的embedding是训练出来的，所以总的结构类似于跳字模型或者连续词袋模型，具体可看[跳字模型](https://zlkqz.top/2022/02/20/NLP%E5%9F%BA%E7%A1%80/#1-2-%E8%B7%B3%E5%AD%97%E6%A8%A1%E5%9E%8B%EF%BC%88skip-gram%EF%BC%89)中的具体实现，简单来说就是：一个单隐藏层的全连接网络，输入one-hot向量，乘一个V矩阵，得到隐藏层值，再乘一个U矩阵，得到输入层值，再softmax计算概率最后梯度下降。**而Decoder的前后就是分别为乘V和乘U两个操作，分别称为embedding转换和pre-softmax linear transformation**
- 在一般的词嵌入模型当中，U、V矩阵一般是两个不同的矩阵，而Transformer中使用了**Weight Tying**，即U、V使用同一矩阵**（注意只是共用权重矩阵，偏差还是相互独立的）**
- one-hot向量和对U的操作是“指定抽取”，即取出某个单词的向量行；pre-softmax对V的操作是“逐个点积”，对隐层的输出，依次计算其和每个单词向量行的变换结果。虽然具体的操作不同，**但在本质上，U和V都是对任一的单词进行向量化表示，然后按词表序stack起来。因此，两个权重矩阵在语义上是相通的**。
- 也是由于上面两种操作方式的不同，且V的更新在靠近输出层，所以**U在反向传播中不如V训练得充分**，将两者绑定在一起缓和了这一问题，可以训练得到质量更高的新矩阵。并且**Weight Tying 可以显著减小模型的参数量**。
- 在embdding层中，**为了让embedding层的权重值不至于过小，乘以$$\sqrt{d_{model} }$$后与位置编码的值差不多，可以保护原有向量空间不被破坏**。



### 2.6 Positional Encode

- 由于模型摒弃了RNN结构，所以**无法获得序列的位置信息**，而为了获得这种位置信息我们需要引入Positional Embedding来表示位置信息。
- Positional Embedding的维度同样是$$d_{model}$$，并且在一开始的时候和Embedding进行相加，具体表示为：

$$
\begin{aligned}
P E_{(p o s, 2 i)} &=\sin \left(p o s / 10000^{2 i / d_{\text {model } }}\right) \\
P E_{(p o s, 2 i+1)} &=\cos \left(p o s / 10000^{2 i / d_{\text {model } }}\right)
\end{aligned}
$$

其中pos代表第几个序列位置（最大值为规定的最长序列长度），i代表第几个维度（最大值为$$d_{mdoel} / 2$$）

- 以上公式不仅能很好的表示单词的绝对位置，还能表示出相对位置：**相隔 k 个词的两个位置 pos 和 pos+k 的位置编码是由 k 的位置编码定义的一个线性变换**：

$$
\begin{array}{c}
P E(p o s+k, 2 i)=P E(p o s, 2 i) P E(k, 2 i+1)+P E(p o s, 2 i+1) P E(k, 2 i) \\
P E(p o s+k, 2 i+1)=P E(p o s, 2 i+1) P E(k, 2 i+1)-P E(p o s, 2 i) P E(k, 2 i)
\end{array}
$$

- 采用正弦方式和学习方式position embedding结果几乎一样。但采用正弦，因为**能让模型推断出比训练期间遇到的序列长度更长的序列长度**





# 3 模型训练

### 3.1 Optimizer && learning rate

- 采用Adam优化器，参数都是模型参数：$$\beta_1=0.9,\beta_2=0.98,\epsilon=10^{-9}$$

- Transformer 的学习率更新公式叫作“**noam**”，它将 warmup 和 decay 两个部分组合在一起，总体趋势是**先增加后减小**，具体公式为：

$$
\text { lrate }=d_{\text {model } }^{-0.5} \cdot \min \left(\text { step }_{-} \text {num }^{-0.5}, \text { step_ }_{-} \text {num } \cdot \text { warmup_steps }^{-1.5}\right)
$$

- 公式实际上是一个以warmup_steps为分界点的分段函数。该点之前是warmup部分，采用线性函数的形式，且warmup_steps越大，斜率越小。该点之后是decay部分，采用负幂的衰减形式，衰减速度先快后慢：

<img src="https://zlkqzimg-1310374208.cos.ap-chengdu.myqcloud.com/v2-7c98b2c7ca4467ab770da064bb2b58ba_720w.jpg" alt="v2-7c98b2c7ca4467ab770da064bb2b58ba_720w"  />

- **设置warmup的原因：**在CV领域中常常这样做，在《Deep Residual Learning for Image Recognition》中，作者训练110层的超深网络是就用过类似策略：

> In this case, we find that **the initial learning rate of 0.1 is slightly too large to start converging**. So we use 0.01 to warm up the training until the training error is below 80% (about 400 iterations), and then go back to 0.1 and continue training.

对于 Transformer 这样的大型网络，**在训练初始阶段，模型尚不稳定，较大的学习率会增加收敛难度**。因此，使用较小的学习率进行 warmup，等 loss 下降到一定程度后，再恢复回常规学习率。



### 3.2 Dropout

- 在每个子块中，输出结果加入到残差结构和layer normalization之前，进行Dropout
- 并且还在Encoder和Decoder最开始的两种embedding相加的时候，使用了Dropout
- Dropout的概率均为0.1



### 3.3 Label Smoothing

- 为了不要对正确类别"too confident"（防止过拟合），Transformer中还使用了Label Smoothing。这种方法**会增大困惑度（perplexity），但是可以提高accuracy和BLEU**。
- 假设目标类别为y，任意类别为k，ground-truth 分布为q(k)，模型预测分布为p(k)。 显然，当k=y时，q(k)=1。当k$$\neq$$y时，q(k)=0。**LSR（Label Smoothing Regularization）为了让模型的输出不要过于贴合单点分布，选择在gound-truth中加入噪声**。即削弱y的概率，并整体叠加一个独立于训练样例的均匀分布u(k)：

$$
q^{\prime}(k)=(1-\epsilon) q(k)+\epsilon u(k)=(1-\epsilon) q(k)+\epsilon / K
$$

其中K为softmax的类别数，拆开来看就是：
$$
\begin{array}{ll}
q^{\prime}(k)=1-\epsilon+\epsilon / K, & k=y \\
q^{\prime}(k)=\epsilon / K, & k \neq y
\end{array}
$$
所有类别的概率和仍然是归一的。说白了就是把最高点砍掉一点，多出来的概率平均分给所有人。


