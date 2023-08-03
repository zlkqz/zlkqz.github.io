---
title: HMM和CRF
math: true
date: 2022-6-14
---



# 1 隐马尔可夫模型

- 隐马尔可夫模型（Hidden Markov Model, HMM）常用于序列标注问题，描述由隐藏的马尔科夫链随机生成观测序列的过程，属于**概率图模型**（用图结构来描述变量之间的关系，属于生成式模型）
- HMM属于[贝叶斯网](https://zlkqz.site/2022/09/28/%E8%B4%9D%E5%8F%B6%E6%96%AF%E5%88%86%E7%B1%BB%E5%99%A8/#6-EM%E7%AE%97%E6%B3%95)，其两个基本假设其实就是贝叶斯网的假设：**给定父节点集，贝叶斯网假设每个属性与他的非后裔属性独立**



### 1.1 模型定义

- 设$$Q=  \{q_1, ..., q_N\}$$是所有可能的状态的集合，$$V = \{v_1, ..., v_M\}$$是所有可能的观测的集合。而$$I = (i_1, ..., i_T)$$是状态序列，$$O = (o_1, ..., o_T)$$是对应的观测序列，模型定义了三种参数：

> 1. **状态转移矩阵A：**
>
> $$
> A = [a_{ij}]_{N \times N}
> $$
>
> 其中$$a_{ij}$$指在时刻t处于状态$$q_i$$的条件下转移到在t+1时刻状态为$$q_j$$的概率：
> $$
> a_{ij} = P(i_{t+1}=q_j|i_t=q_i)
> $$
> 2. **观测概率矩阵B：**
>
> $$
> B = [b_j(k)]_{N\times M}
> $$
>
> 其中$$b_j(k)$$指在t时刻处于状态$$q_j$$时生成观测$$v_k$$的概率：
> $$
> b_j(k) = P(o_t = v_k|i_t =  q_j)
> $$
> 3. **初始状态概率向量$$\pi$$：**
>
> $$
> \pi = (\pi_i)
> $$
>
> 其中$$\pi_i$$是$$t=1$$时处于状态$$q_i$$的概率：
> $$
> \pi = P(i_1 = q_i)
> $$

- 一般使用一个三元组表示HMM的参数：

$$
\lambda = (A,B,\pi)
$$

- 上面对参数的定义中，隐含了HMM的两个基本假设：

> 1. **齐次马尔可夫性假设：**假设隐藏的马尔可夫链在任意时刻t的状态只依赖于前一时刻的状态，与其他时刻的状态及观测无关，也与时刻t无关：
>
> $$
> P\left(i_{t} \mid i_{t-1}, o_{t-1}, \cdots, i_{1}, o_{1}\right)=P\left(i_{t} \mid i_{t-1}\right), \quad t=1,2, \cdots, T
> $$
>
> 2. **观测独立性假设：**假设任意时刻的观测只依赖于该时刻的马尔可夫链的状态，与其他观测及状态无关：
>
> $$
> P\left(o_{t} \mid i_{T}, o_{T}, i_{T-1}, o_{T-1}, \cdots, i_{t+1}, o_{t+1}, i_{t}, i_{t-1}, o_{t-1}, \cdots, i_{1}, o_{1}\right)=P\left(o_{t} \mid i_{t}\right)
> $$
>
> **其实这两个假设就是贝叶斯网的假设，只不过结构特殊一点，是一个线性结构**





### 1.2 概率计算

- 概率计算即给定模型$$\lambda = (A,B,\pi)$$和观测序列$$O=(o_1, ..., o_T)$$，计算该观测序列出现的概率$$P(O|\lambda)$$



#### 1.2.1 直接计算

- 直接计算即列举所有可能的状态序列$$I$$，计算：

$$
\begin{aligned}
P(O \mid \lambda) &=\sum_{I} P(O \mid I, \lambda) P(I \mid \lambda) \\
&=\sum_{i_{1}, i_{2}, \cdots, i_{T}} \pi_{i_{1}} b_{i_{1}}\left(o_{1}\right) a_{i_{1} i_{2}} b_{i_{2}}\left(o_{2}\right) \cdots a_{i_{T-i} i_{T}} b_{i_{T}}\left(o_{T}\right)
\end{aligned}
$$

- 但是这种方法计算量过大，复杂度为$$O(TN^T)$$，所以是不可行的



#### 1.2.2 前向计算

- 给定模型$$\lambda$$，定义在时刻t时观测序列为$$o_1, ..., o_t$$，且当前状态为$$q_i$$的概率为前向概率：

$$
\alpha_t(i) = P(o_1, ..., o_t, i_t=q_i|\lambda)
$$

- 算法流程：

给定模型$$\lambda$$和观测序列$$O=(o_1, ..., o_T)$$

<img src="https://zlkqzimg-1310374208.cos.ap-chengdu.myqcloud.com/image-20221028120947404.png" alt="image-20221028120947404" style="zoom:80%;" />



#### 1.2.3 后向计算

- 给定模型$$\lambda$$，定义在时刻t时刻状态为$$q_i$$的条件下，从t+1到T的部分观测序列为$$o_{t+1}, ..., o_T$$的概率为后向概率：

$$
\beta_{t}(i)=P\left(o_{t+1}, o_{t+2}, \cdots, o_{T} \mid i_{t}=q_{i}, \lambda\right)
$$

- 算法流程：

<img src="https://zlkqzimg-1310374208.cos.ap-chengdu.myqcloud.com/image-20221028122410576.png" alt="image-20221028122410576" style="zoom:75%;" />



#### 1.2.4 常用概率的计算

1. **计算给定模型$$\lambda$$和观测$$O$$，在t时刻处于$$q_i$$的概率：**

$$
\gamma_t(i) = P(i_t = q_i|O, \lambda)
$$

> - 首先运用贝叶斯公式：
>
> $$
> \gamma_{t}(i)=P\left(i_{t}=q_{i} \mid O, \lambda\right)=\frac{P\left(i_{t}=q_{i}, O \mid \lambda\right)}{P(O \mid \lambda)}
> $$
>
> - 由上述的前向后向概率的定义可得：
>
> $$
> \alpha_t(i)\beta_t(i) = P(i_t=q_i,O|\lambda)
> $$
>
> - 于是得到：
>
> $$
> \gamma_{t}(i)=\frac{\alpha_{t}(i) \beta_{t}(i)}{P(O \mid \lambda)}=\frac{\alpha_{t}(i) \beta_{t}(i)}{\sum_{j=1}^{N} \alpha_{t}(j) \beta_{t}(j)}
> $$

2. **计算给定模型$$\lambda$$和观测$$O$$，在t时刻处于状态$$q_i$$并且在t+1时刻处于状态$$q_j$$的概率：**

$$
\xi_{t}(i, j)=P\left(i_{t}=q_{i}, i_{t+1}=q_{j} \mid O, \lambda\right)
$$

> - 同样先运用贝叶斯公式：
>
> $$
> \xi_{t}(i, j)=\frac{P\left(i_{t}=q_{i}, i_{t+1}=q_{j}, O \mid \lambda\right)}{P(O \mid \lambda)}=\frac{P\left(i_{t}=q_{i}, i_{t+1}=q_{j}, O \mid \lambda\right)}{\sum_{i=1}^{N} \sum_{j=1}^{N} P\left(i_{t}=q_{i}, i_{t+1}=q_{j}, O \mid \lambda\right)}
> $$
>
> - 其中：
>
> $$
> P\left(i_{t}=q_{i}, i_{t+1}=q_{j}, O \mid \lambda\right)=\alpha_{t}(i) a_{i j} b_{j}\left(o_{t+1}\right) \beta_{t+1}(j)
> $$
>
> - 所以：
>
> $$
> \xi_{t}(i, j)=\frac{\alpha_{t}(i) a_{i j} b_{j}\left(o_{t+1}\right) \beta_{t+1}(j)}{\sum_{i=1}^{N} \sum_{j=1}^{N} \alpha_{t}(i) a_{i j} b_{j}\left(o_{t+1}\right) \beta_{t+1}(j)}
> $$



### 1.3 学习算法

- 在观测序列和状态序列都给定的时候，训练集足够大时，可以直接通过极大似然估计来估计模型参数，即**直接通过频数来估计概率**
- 我们主要讨论的是只给定观测序列时的情况，**此时状态序列为隐变量**，所以使用[EM算法](https://zlkqz.site/2022/09/28/%E8%B4%9D%E5%8F%B6%E6%96%AF%E5%88%86%E7%B1%BB%E5%99%A8/#6-EM%E7%AE%97%E6%B3%95)

- 首先是E步，即计算Q函数：

$$
Q(\lambda, \bar{\lambda}) = \sum_{I}\log P(O,I|\lambda)P(I|O,\bar{\lambda}) = \sum_{I}\log P(O,I|\lambda)\frac{P(O,I|\bar{\lambda})}{P(O|\bar{\lambda})}
$$

其中$$\bar{\lambda}$$是HMM参数的当前估计值，$$\lambda$$是要极大化的值，作为下次迭代的新参数

- 由于分母中的$$P(O|\bar{\lambda})$$和要更新的参数$$\lambda$$无关，所以直接去掉，Q函数直接化为：

$$
Q(\lambda, \bar{\lambda})=\sum_{I} \log P(O, I \mid \lambda) P(O, I \mid \bar{\lambda})
$$

- 其中：

$$
P(O, I \mid \lambda)=\pi_{i_{1}} b_{i_{1}}\left(o_{1}\right) a_{i_{i} i_{2}} b_{i_{2}}\left(o_{2}\right) \cdots a_{i_{T-1} i_{T}} b_{i_{T}}\left(o_{T}\right)
$$

- 所以：

$$
\begin{aligned}
Q(\lambda, \bar{\lambda})=& \sum_{I} \log \pi_{i} P(O, I \mid \bar{\lambda}) +\sum_{I}\left(\sum_{t=1}^{T-1} \log a_{i, i_{i}+1}\right) P(O, I \mid \bar{\lambda})+\sum_{I}\left(\sum_{t=1}^{T} \log b_{i_{i}}\left(o_{t}\right)\right) P(O, I \mid \bar{\lambda})
\end{aligned}
$$

- 然后是M步，即最大化Q函数，而最大化Q函数的问题，可以化为分别最大化上式中Q函数中的三项：

> 1. 最大化$$\sum_{I} \log \pi_{i} P(O, I \mid \bar{\lambda})$$，可以先将其化为：
>
> $$
> \sum_{I} \log \pi_{i_{0}} P(O, I \mid \bar{\lambda})=\sum_{i=1}^{N} \log \pi_{i} P\left(O, i_{1}=i \mid \bar{\lambda}\right)
> $$
>
> 由于存在约束$$\sum_{i=1}^N\pi_i =1$$，所以可以使用拉格朗日乘子法进行求解，最后得到：
> $$
> \pi_{i}=\frac{P\left(O, i_{1}=i \mid \bar{\lambda}\right)}{P(O \mid \bar{\lambda})}
> $$
>
> 2. 最大化$$\sum_{I}\left(\sum_{t=1}^{T-1} \log a_{i, i_{i}+1}\right) P(O, I \mid \bar{\lambda})$$，化为：
>
> $$
> \sum_{I}\left(\sum_{t=1}^{T-1} \log a_{i_{i} i_{t+1}}\right) P(O, I \mid \bar{\lambda})=\sum_{i=1}^{N} \sum_{j=1}^{N} \sum_{t=1}^{T-1} \log a_{i j} P\left(O, i_{t}=i, i_{t+1}=j \mid \bar{\lambda}\right)
> $$
>
> 运用拉格朗日乘子法，最后得到：
> $$
> a_{i j}=\frac{\sum_{t=1}^{T-1} P\left(O, i_{t}=i, i_{t+1}=j \mid \bar{\lambda}\right)}{\sum_{t=1}^{T-1} P\left(O, i_{t}=i \mid \bar{\lambda}\right)}
> $$
>
> 3. 最大化$$\sum_{I}\left(\sum_{t=1}^{T} \log b_{i_{i}}\left(o_{t}\right)\right) P(O, I \mid \bar{\lambda})$$，化为：
>
> $$
> \sum_{I}\left(\sum_{t=1}^{T} \log b_{i_{i}}\left(o_{t}\right)\right) P(O, I \mid \bar{\lambda})=\sum_{j=1}^{N} \sum_{t=1}^{T} \log b_{j}\left(o_{t}\right) P\left(O, i_{t}=j \mid \bar{\lambda}\right)
> $$
>
> 运用拉格朗日乘子法，最后得到：
> $$
> b_{j}(k)=\frac{\sum_{t=1}^{T} P\left(O, i_{t}=j \mid \bar{\lambda}\right) I\left(o_{t}=v_{k}\right)}{\sum_{t=1}^{T} P\left(O, i_{t}=j \mid \bar{\lambda}\right)}
> $$

- 算法流程：

<img src="https://zlkqzimg-1310374208.cos.ap-chengdu.myqcloud.com/image-20221028132201642.png" alt="image-20221028132201642" style="zoom:67%;" />



### 1.4 预测算法

- 序列模型常用**维特比算法**来进行预测，算法流程如下：

<img src="https://zlkqzimg-1310374208.cos.ap-chengdu.myqcloud.com/image-20221028132500929.png" alt="image-20221028132500929" style="zoom:67%;" />





# 2 条件随机场

- 条件随机场（Conditional Random Field, CRF）和HMM一样，同样属于概率图模型，是给定一组输入随机变量条件下，另一组输出是随机变量的条件概率分布模型
- 有一种特殊且最常用的CRF为线性链条件随机场，其结构和数学推导和HMM十分相似，但是仍有一些区别，稍后介绍



### 2.1 马尔可夫随机场

- **概率图模型**是由图结构来描述变量之间的关系的模型，而采用有向无环图结构被称为**贝叶斯网**，采用无向图则被称为**概率无向图模型或马尔可夫随机场**

- 设有联合概率分布$$P(Y)$$，$$Y \in \mathcal{Y}$$是一组随机变量。由无向图$$G = (V,E)$$表示概率分布$$P(Y)$$，即在G中，结点$$v \in V$$表示一个随机变量$$Y_v$$，$$Y = (Y_v)_{v \in V}$$，而边$$e \in E$$表示随机变量间的概率依赖关系
- 首先定义无向图模型内的马尔可夫性：

> 1. **成对马尔可夫性：**设u和v是图中任意两个没有边连接的结点，其对应的随机变量分别为$$Y_u, Y_v$$，其他所有结点为O，对应随机变量$$Y_O$$。成对马尔可夫性指给定$$Y_O$$的条件下，$$Y_u, Y_v$$是条件独立的：
>
> $$
> P\left(Y_{u}, Y_{v} \mid Y_{o}\right)=P\left(Y_{u} \mid Y_{o}\right) P\left(Y_{v} \mid Y_{o}\right)
> $$
>
> 2. **局部马尔可夫性：**设v为图中任意一个结点，W是与v相连的所有结点，O是除v和W之外的所有结点。局部马尔可夫性指给定$$Y_W$$的条件下$$Y_v, Y_O$$是条件独立的：
>
> $$
> P\left(Y_{v}, Y_{o} \mid Y_{W}\right)=P\left(Y_{v} \mid Y_{W}\right) P\left(Y_{o} \mid Y_{W}\right)
> $$
>
> 也可以为表示为：
> $$
> P(Y_v|Y_W) = P(Y_v|Y_W, Y_O)
> $$
>
> 3. **全局马尔可夫性：**A、B是在图中被结点集合C分开的任意结点集合（如下图所示）。全局马尔可夫性指给定$$Y_C$$条件下$$Y_A,Y_B$$条件独立：
>
> $$
> P\left(Y_{A}, Y_{B} \mid Y_{C}\right)=P\left(Y_{A} \mid Y_{C}\right) P\left(Y_{B} \mid Y_{C}\right)
> $$
>
> <img src="https://zlkqzimg-1310374208.cos.ap-chengdu.myqcloud.com/image-20221029144742544.png" alt="image-20221029144742544" style="zoom:67%;" />
>
> **上述三种马尔可夫性是等价的**

- 而马尔可夫随机场，不仅要满足使用无向图，还需要满足马尔可夫性。显然，CRF属于马尔可夫随机场



### 2.2 团和极大团

- 无向图中的任意一个强连通子集都称为**团（clique）**，而一个团不能再加任意一个结点使其仍为团，则这种团为**极大团（maximal clique）**

- 联合概率分布可以用每个团的**势函数（potential function）**的乘积来表示，但是一个图中的团很多，且有些随机变量同时属于多个团，所以简化来说，可以直接使用**极大团的势函数成绩**：

$$
P(Y)=\frac{1}{Z} \prod_{C} \Psi_{C}\left(Y_{C}\right)
$$

其中C为极大团集，Z为规范化因子，$$\Psi$$为势函数，要求势函数是严格正的，一般定义为指数函数：
$$
\Psi_{C}\left(Y_{C}\right)=\exp \left\{-E\left(Y_{C}\right)\right\}
$$


### 2.3 模型定义

- 设随机变量$$X,Y$$，如果对任意结点v满足马尔可夫性（下式为局部马尔可夫性）：

$$
P\left(Y_{v} \mid X, Y_{w}, w \neq v\right)=P\left(Y_{v} \mid X, Y_{w}, w \sim v\right)
$$

则称条件概率$$P(Y|X)$$为条件随机场

- 另外有CRF的特例：**线性链条件随机场**，满足马尔可夫性：

$$
\begin{array}{c}
P\left(Y_{i} \mid X, Y_{1}, \cdots, Y_{i-1}, Y_{i+1}, \cdots, Y_{n}\right)=P\left(Y_{i} \mid X, Y_{i-1}, Y_{i+1}\right) \\
i=1,2, \cdots, n \text { (在 } i=1 \text { 和 } n \text { 时只考虑单边) }
\end{array}
$$

则称条件概率$$P(Y|X)$$为条件随机场，图结构如下：

<img src="https://zlkqzimg-1310374208.cos.ap-chengdu.myqcloud.com/image-20221029152606453.png" alt="image-20221029152606453" style="zoom: 67%;" />

- CRF定义中没有要求X的结构，但是一般假设X和Y有相同的图结构，比如线性链条件随机场：

<img src="https://zlkqzimg-1310374208.cos.ap-chengdu.myqcloud.com/image-20221029152749755.png" alt="image-20221029152749755" style="zoom:67%;" />

在标注问题中（如NER），X表示输入观测序列，Y表示对应的输出标记序列或状态序列



### 2.4 CRF的参数

- 现在开始介绍的CRF都默认为线性链CRF
- 前面说过，马尔科夫场的概率可以用极大团的势函数来表示：

$$
P(Y)=\frac{1}{Z} \prod_{C} \Psi_{C}\left(Y_{C}\right) = \frac{1}{Z}\exp \sum_C-E\left(Y_{C}\right) = \frac{1}{Z}\exp \sum_CF_C\left(Y_{C}\right)
$$

- 对于线性链CRF，每一个$$y_{i-1}, y_i$$构成一个极大团，所以：

$$
P(Y|X) = \frac{1}{Z}\exp \sum_{t=1}^TF_t(y_{t-1},y_t,x)
$$

其中T为时间步数，$$x$$为X的取值，是整个观测序列。**由于每个极大团的结构都相同**，所以可以**简化为每个极大团的势函数都一样**：
$$
P(Y|X) = \frac{1}{Z}\exp \sum_{t=1}^TF(y_{t-1},y_t,x)
$$

- 而对于每个$$F(y_{t-1},y_t,x)$$，可以表示为三个函数的和：

$$
F(y_{t-1},y_t,x) = F_1(y_{t-1}, x) + F_1(y_t, x) + F_2(y_{t-1}, y_t, x)
$$

其中$$F_1$$就称为**状态函数**，$$F_2$$称为**转移函数**。由于$$F_1(y_{t-1},x)$$在上一个时间步已经出现过了，所以可以直接去掉：
$$
F(y_{t-1},y_t,x) = F_1(y_t, x) + F_2(y_{t-1}, y_t, x)
$$

- 那么可以引入特征函数来定义$$F_1, F_2$$：

$$
F_1 = \sum_{l=1}^{K_2}\mu_ls_l(y_t, x) \\
F_2 = \sum_{k=1}^{K_1}\lambda_kt_k(y_{t-1}, y_t, x)
$$

- 所以条件概率就表示为：

$$
P(Y|X) = \frac{1}{Z}\exp \sum_{t=1}^T(\sum_{k=1}^{K_1}\lambda_kt_k(y_{t-1}, y_t, x) + \sum_{l=1}^{K_2}\mu_ls_l(y_t, x))
$$

- 而可以把关于时间步的求和放入括号中：

$$
P(Y|X) = \frac{1}{Z}\exp(\sum_{k=1}^{K_1}\lambda_k\sum_{t=1}^Tt_k(y_{t-1}, y_t, x)+\sum_{l=1}^{K_2}\mu_l\sum_{t=1}^Ts_l(y_t, x))
$$

- 将两种特征函数和其权重合起来：

$$
f_{k}\left(y_{t-1}, y_{t}, x\right)=\left\{\begin{array}{l}
t_{k}\left(y_{t-1}, y_{t}, x\right), \quad k=1,2, \cdots, K_{1} \\
s_{l}\left(y_{t}, x\right), \quad k=K_{1}+l ; l=1,2, \cdots, K_{2}
\end{array}\right.  \\
w_{k}=\left\{\begin{array}{ll}
\lambda_{k}, & k=1,2, \cdots, K_{1} \\
\mu_{l}, & k=K_{1}+l ; l=1,2, \cdots, K_{2}
\end{array}\right.
$$

然后对特征函数在各个时间步进行求和，记作：
$$
f_{k}(y, x)=\sum_{t=1}^{T} f_{k}\left(y_{t-1}, y_{t}, x\right), \quad k=1,2, \cdots, K
$$

- 所以条件概率化简为：

$$
P(y \mid x)=\frac{1}{Z(x)} \exp \sum_{k=1}^{K} w_{k} f_{k}(y, x)
$$

- 上式采用向量化表示，引入：

$$
\begin{array}{c}
w=\left(w_{1}, w_{2}, \cdots, w_{K}\right)^{\mathrm{T}} \\
F(y, x)=\left(f_{1}(y, x), f_{2}(y, x), \cdots, f_{K}(y, x)\right)^{\mathrm{T}}
\end{array}
$$

所以条件概率的向量化表示为：
$$
\begin{array}{l}
P_{w}(y \mid x)=\frac{\exp (w \cdot F(y, x))}{Z_{w}(x)} \\
Z_{w}(x)=\sum_{y} \exp (w \cdot F(y, x))
\end{array}
$$
**其中特征函数$$F(y,x)$$是事先设计好给出的，而学习的目标即学习权重$$w$$，最大化特征函数的总得分**



### 2.5 概率计算

- 和前面HMM一样，同样运用前向后向算法的变量来进行概率计算，首先定义一个矩阵，**为了方便讨论，我们又引入了两个时间步的状态序列$$y_0 = start, y_{T+1} = stop$$，实际前文的讨论中也隐含了$$y_0 = start$$：**

$$
\begin{array}{c}
M_{t}(x)=\left[M_{t}\left(y_{t-1}, y_{t} \mid x\right)\right] \\
M_{t}\left(y_{t-1}, y_{t} \mid x\right)=\exp \sum_{k=1}^{K} w_{k} f_{k}\left(y_{t-1}, y_{t}, x\right)
\end{array}
$$

设$$y_t$$有m个取值，则矩阵$$M_t(x)$$里面的$$y_t,y_{t-1}$$分别取不同的m个值，所以$$M_t(x)$$是一个$$m \times m$$阶矩阵

- 有了上述定义，可以将条件概率进一步写为矩阵形式：

$$
P_{w}(y \mid x)=\frac{1}{Z_{w}(x)} \prod_{t=1}^{T} M_{t}\left(y_{t-1}, y_{t} \mid x\right)
$$

其中$$Z_w(x)$$是T+1个矩阵乘积的第(start, end)元素：
$$
Z_{w}(x)=\left(M_{1}(x) M_{2}(x) \cdots M_{T+1}(x)\right)_{\text {start,stop }}
$$

- 现在来定义**前向概率，$$\alpha_t(y_t|x)$$为在时刻t时观测序列为$$x_1, ..., x_t$$，且当前状态为$$y_t$$的概率**：

$$
\alpha_{t}\left(y_{t} \mid x\right)=\alpha_{t-1}\left(y_{t-1} \mid x\right) M_{t}\left(y_{t-1}, y_{t} \mid x\right) \quad t=1,...,T+1 \\
\alpha_{0}(y \mid x)=\left\{\begin{array}{ll}
1, & y=\operatorname{start} \\
0, & \text { 否则 }
\end{array}\right.
$$

因为$$y_t$$有m个取值，所以可以定义m维向量$$\alpha_t(x)$$：
$$
\alpha_{t}^{\mathrm{T}}(x)=\alpha_{t-1}^{\mathrm{T}}(x) M_{t}(x) \\
\alpha_0(x) = \mathbb{1}（单位向量）
$$

- 同样可以定义**后向概率，$$\beta_t(y_t|x)$$为在时刻t观测序列为$$x_{t+1}, ..., x_T$$，且当前状态为$$y_t$$的概率：**

$$
\beta_{t}\left(y_{t} \mid x\right)=M_{t}\left(y_{t}, y_{t+1} \mid x\right) \beta_{t+1}\left(y_{t+1} \mid x\right) \\
\beta_{t+1}\left(y \mid x\right)=\left\{\begin{array}{ll}
1, & y=\text { stop } \\
0, & \text { 否则 }
\end{array}\right.
$$

同样有向量形式：
$$
\beta_{t}(x)=M_{t+1}(x) \beta_{t+1}(x) \\
\beta_{T+1}(x) = \mathbb{1}  (单位向量)
$$

- 得到了前后向概率，就可以得到**在时刻t状态是$$y_t$$的概率：**

$$
P\left(Y_{t}=y_{t} \mid x\right)=\frac{\alpha_{t}\left(y_{t} \mid x\right) \beta_{t}\left(y_{t} \mid x\right)}{Z(x)}
$$

- 以及**在时刻t-1状态是$$y_{t-1}$$并且在时刻t状态是$$y_t$$的概率：**

$$
P\left(Y_{t-1}=y_{t-1}, Y_{t}=y_{t} \mid x\right)=\frac{\alpha_{t-1}\left(y_{t-1} \mid x\right) M_{t}\left(y_{t-1}, y_{t} \mid x\right) \beta_{t}\left(y_{t} \mid x\right)}{Z(x)}
$$

- 上述两个式子的$$Z(x)$$可以用更简单的形式表达：

$$
Z(x) = \alpha_n^T(x)\mathbb{1} = \mathbb{1}\beta_1(x)
$$



### 2.6 学习和预测

- 预测算法和HMM一样，使用**维特比算法**

- 学习算法其实就是[最大熵模型的学习](https://zlkqz.site/2022/10/27/%E6%9C%80%E5%A4%A7%E7%86%B5%E6%A8%A1%E5%9E%8B/#3-%E4%BC%98%E5%8C%96%E7%AE%97%E6%B3%95)





# 3 HMM和CRF的区别

- CRF和HMM的数学推导几乎是一样的，但是差异在于：

> 1. HMM属于贝叶斯网络，而CRF属于马尔可夫场，前者的假设和约束更加的严格，比如CRF并没有像HMM一样完全依赖上一步的状态，HMM的观测变量的生成是独立的
> 2. 上述我们讨论的仅仅只是线性链CRF，但是CRF还可以有其他的图结构，并且**实际上可以任意选定特征函数的个数和形式，特征函数的不确定也是CRF能和深度学习融合的最主要原因，模型可以自己学习特征函数，并且不用显示地表达出来**





# 4 CRF和深度学习模型的结合

- 以BiLSTM+CRF做NER任务举例

- 如果不用CRF而是直接在模型的后面接一个Softmax，鉴于所选取的基模型的强大的特征抽取能力，这已经可以有比较好的分类效果，**但是NER任务是存在一些约束的**，比如BIO格式中，B-Person后面不可能跟I-Organization。**Softmax的分类是每个时间步相互独立的，所以可能会出现上述的问题**

- **而CRF层可以加入一些约束来保证最终预测结果是有效的，这些约束可以在训练数据时被CRF层自动学习得到**

- 首先介绍一下模型结构：

<img src="https://zlkqzimg-1310374208.cos.ap-chengdu.myqcloud.com/v2-0650e1511e7d4419c9528a8d08ea61fd_720w.webp" alt="img" style="zoom:80%;" />

BiLSTM的输出经过Dense层，**转化为每个Label对应的score（图中浅黄色部分），这个score即CRF中的状态得分$$\sum \mu s$$**，然后将其输入CRF，CRF层维护了一个转移矩阵（Transition Matrix），这也是CRF层中需要学习的参数，假设总共有包括START和END在内的7个label，则转移矩阵为：

<img src="https://zlkqzimg-1310374208.cos.ap-chengdu.myqcloud.com/v2-2064e34cece3be4e852b1ace6bbca2ba_720w.webp" alt="img" style="zoom:67%;" />

**每个元素代表了相邻时间步之间进行状态转移的score，这个score即CRF中的转移得分$$\sum\lambda t$$**，并且上述矩阵已经学到了一些有用的约束：

> 1. 句子的第一个单词应该是“B-” 或 “O”，而不是“I”。（从“START”->“I-Person 或 I-Organization”的转移分数很低）
> 2. “B-label1 I-label2 I-label3…”，在该模式中，类别1,2,3应该是同一种实体类别。比如，“B-Person I-Person” 是正确的，而“B-Person I-Organization”则是错误的。（“B-Organization” -> “I-Person”的分数很低）
> 3. O I-label”是错误的，命名实体的开头应该是“B-”而不是“I-”

- 每个输入有N中可能的结果，即N条路径，比如（加粗的为真实路径）：

> 1. START B-Person B-Person B-Person B-Person B-Person END
> 2. START B-Person I-Person B-Person B-Person B-Person END
> 3. **START B-Person I-Person O B-Organization O END**
>
> ......
>
>    N. O O O O O O O

- 训练目标即最大化真实路径的得分，可得损失函数：

$$
Loss
 =-\log \frac{P_{\text {Real Path }}}{P_{1}+P_{2}+\ldots+P_{N}} 
 =-\log \frac{e^{s_{\text {Realpath }}}}{e^{s_{1}+e^{s_{2}}+\ldots+e^{s_{N}}}}  \\
 =-\left(S_{\text {RealPath }}-\log \left(e^{S_{1}}+e^{S_{2}}+\ldots+e^{S_{N}}\right)\right)
$$

其中$$S_i$$为一条路径对应的得分，是通过Softmax实现最大化的

- 值得一提的是，计算分母中的所有路径的得分和$$-\log (e^{S_1} + ... + e^{S_N})$$不需要列举所有可能路径，可以用一种动态规划的方法降低计算复杂度

- 另外，在进行预测的时候，同样是使用维特比算法