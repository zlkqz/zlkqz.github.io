---
title: SVM总结
math: true
date: 2022-4-6
---



- SVM和决策树一样，同样是一种判别式模型， 都是基于条件概率分布进行建模，要在样本空间中找到一个划分超平面，将不同类别的样本分开，如下图：

<img src="https://zlkqzimg-1310374208.cos.ap-chengdu.myqcloud.com/image-20220923114248027.png" alt="image-20220923114248027" style="zoom:67%;" />

存在多个超平面可将训练样本分开，但是我们是希望找到对于分类结果**最鲁棒的超平面**，也就是图中加粗的那条线，这个超平面对训练样本局部扰动的“容忍性”最好



# 1 基本概念

- 在样本空间中，一个超平面可以通过以下线性方程来描述：

$$
\boldsymbol{w}^{\mathrm{T}} \boldsymbol{x}+b=0
$$

其中$$w = (w_1, ..., w_d)$$为法向量，确定超平面的方向，b为位移项，确定超平面与原点之间的距离。

- 那么样本空间中任意点$$x$$到超平面$$(w, b)$$的距离为：

$$
r=\frac{\left|\boldsymbol{w}^{\mathrm{T}} \boldsymbol{x}+b\right|}{\|\boldsymbol{w}\|}
$$

- 若超平面$$(w, b)$$能将样本正确分类，则对于任意$$(x_i, y_i) \in D$$，若$$y = +1$$则$$w^Tx_i + b > 0$$，若$$y = -1$$，则$$w^Tx_i + b < 0$$，那么令：

$$
\left\{\begin{array}{ll}
\boldsymbol{w}^{\mathrm{T}} \boldsymbol{x}_{i}+b \geqslant+1, & y_{i}=+1 \\
\boldsymbol{w}^{\mathrm{T}} \boldsymbol{x}_{i}+b \leqslant-1, & y_{i}=-1
\end{array}\right.
$$

另上式等号成立的样本点成为**支持向量**，两个异类支持向量到超平面的距离之和为：
$$
\gamma = \frac{2}{||w||}
$$
称之为**间隔**，如下图：

<img src="https://zlkqzimg-1310374208.cos.ap-chengdu.myqcloud.com/image-20220923120639393.png" alt="image-20220923120639393" style="zoom:75%;" />

> 可以发现在上式中，$$w^Tx_i + b$$是采用了大于或小于$$\pm 1$$而不是0或其他数字进行划分。**首先不能使用0，因为若为0，则支持向量会恰好落在划分超平面上。而使用$$\pm1$$只是因为方便计算（若为其他非0数是一样的效果，因为$w$和$b$可以进行放缩）**

- 欲找到一个具有**最大间隔**的划分超平面，也就是找到能满足约束的参数$w$和$b$，使得$\gamma$最大，即：

$$
\begin{aligned}
\max _{\boldsymbol{w}, b} & \frac{2}{\|\boldsymbol{w}\|} \\
\text { s.t. } & y_{i}\left(\boldsymbol{w}^{\mathrm{T}} \boldsymbol{x}_{i}+b\right) \geqslant 1, \quad i=1,2, \ldots, m .
\end{aligned}
$$

- 因为最大化$$||w||^{-1}$$等价于最小化$$||w||^2$$，所以我们一般写为：

$$
\begin{aligned}
\min _{\boldsymbol{w}, b} & \frac{1}{2}\|\boldsymbol{w}\|^{2} \\
\text { s.t. } & y_{i}\left(\boldsymbol{w}^{\mathrm{T}} \boldsymbol{x}_{i}+b\right) \geqslant 1, \quad i=1,2, \ldots, m .
\end{aligned}
$$

这就是SVM的基本型





# 2 对偶问题

### 2.1 KKT条件

- 考虑一个有m个等式约束和n个不等式约束的优化问题：

$$
\begin{array}{ll}
\min _{\boldsymbol{x}} & f(\boldsymbol{x}) \\
\text { s.t. } & h_{i}(\boldsymbol{x})=0 \quad(i=1, \ldots, m) \\
& g_{j}(\boldsymbol{x}) \leqslant 0 \quad(j=1, \ldots, n)
\end{array}
$$

- 引入拉格朗日乘子$$\pmb{\lambda} = (\lambda_1, ..., \lambda_m)$$和$$\pmb{\mu} = (\mu_1, ..., \mu_n)$$，则相应的拉格朗日函数为：

$$
L(\boldsymbol{x}, \boldsymbol{\lambda}, \boldsymbol{\mu})=f(\boldsymbol{x})+\sum_{i=1}^{m} \lambda_{i} h_{i}(\boldsymbol{x})+\sum_{j=1}^{n} \mu_{j} g_{j}(\boldsymbol{x})
$$

则由不等式约束引入的KKT条件（j = 1, .., n）为：
$$
\left\{\begin{array}{l}
g_{j}(\boldsymbol{x}) \leqslant 0 \\
\mu_{j} \geqslant 0 \\
\mu_{j} g_{j}(\boldsymbol{x})=0
\end{array}\right.
$$



### 2.2 原问题转化到对偶问题

- SVM没有使用原问题而使用对偶问题是因为：**对偶函数更易于求解，并且对偶函数是一个光滑的凸函数，可以找到全局最优解，**具体解释可看[例子](https://www.zhihu.com/question/36694952)

- 先写出上述问题的拉格朗日函数，即**原问题**：

$$
\mathcal{L}(w, b, \alpha)=\frac{1}{2}\|w\|^{2}-\sum_{i=1}^{n} \alpha_{i}\left(y_{i}\left(w^{T} x_{i}+b\right)-1\right)
$$

其中的$$\alpha_i$$为拉格朗日乘子

- 易知：**当有一个约束函数不满足时，L的最大值为$$\infty$$（只需令其对应的$$\alpha_i$$为$$\infty$$即可）；当所有约束条件都满足时，L的最大值为$$\frac{1}{2}||w||^2$$（只需令所有$$\alpha_i$$为0）**。所有原问题等价于：

$$
\min _{\boldsymbol{w}, b} \frac{1}{2}||w||^2 = \min _{w, b} \theta(w)=\min _{w, b} \max _{\alpha_{i} \geq 0} \mathcal{L}(w, b, \alpha)=p^{*}
$$

- 由于这个的求解问题不好做，**因此一般我们将最小和最大的位置交换一下（需满足KKT条件）**：

$$
\max _{\alpha_{i} \geq 0} \min _{w, b} \mathcal{L}(w, b, \alpha)=d^{*}
$$

- 接下来就先对w，b求极小，再对$$\alpha$$求极大：

> 1. 首先求L对w和b的极小，分别求L关于w和b的偏导，可以得出：
>
> $$
> \begin{array}{l}
> \frac{\partial L}{\partial w}=0 \Rightarrow w=\sum_{i=1}^{n} \alpha_{i} y_{i} x_{i} \\
> \frac{\partial L}{\partial b}=0 \Rightarrow \sum_{i=1}^{n} \alpha_{i} y_{i}=0
> \end{array}
> $$
>
> 2. 将上述结果代入L：
>
> $$
> \begin{aligned}
> \mathcal{L}(w, b, \alpha) &=\frac{1}{2} \sum_{i, j=1}^{n} \alpha_{i} \alpha_{j} y_{i} y_{j} x_{i}^{T} x_{j}-\sum_{i, j=1}^{n} \alpha_{i} \alpha_{j} y_{i} y_{j} x_{i}^{T} x_{j}-b \sum_{i=1}^{n} \alpha_{i} y_{i}+\sum_{i=1}^{n} \alpha_{i} \\
> &=\sum_{i=1}^{n} \alpha_{i}-\frac{1}{2} \sum_{i, j=1}^{n} \alpha_{i} \alpha_{j} y_{i} y_{j} x_{i}^{T} x_{j}
> \end{aligned}
> $$

- 这样就得到了原问题的**对偶问题**：

$$
\begin{array}{ll}
\max _{\alpha} & \sum_{i=1}^{n} \alpha_{i}-\frac{1}{2} \sum_{i, j=1}^{n} \alpha_{i} \alpha_{j} y_{i} y_{j} x_{i}^{T} x_{j} \\
\text { s.t. } & \alpha_{i} \geq 0, i=1, \ldots, n \\
& \sum_{i=1}^{n} \alpha_{i} y_{i}=0
\end{array}
$$

然后再对$$\alpha$$求解**（采用SMO算法）**，即可得到模型：
$$
f(x) = w^Tx + b \\  = \sum_{i=1}^m{\alpha_iy_ix_i^Tx + b}
$$

> 由于满足KKT条件，所有对于任意样本$$(x_i, y_i) \in D$$，总有$$\alpha_i=0$$或$$y_if(x_i) = 1$$。若$$\alpha_i=0$$，则样本将不会出现在上述的模型式子中，就不会对$$f(x)$$产生影响；若$$\alpha_i > 0$$，则对应样本为支持向量。所以：**最终模型仅与支持向量有关，大部分训练样本都无需保留**



### 2.3 SMO算法

- 在对偶问题中如果要对$$\alpha$$求解，这是一个二次规划问题，可使用通用的方法求解，**但是该问题的规模正比于训练样本数，将会有很大的开销**，所以提出了SMO（Sequential Minimal Optimization）等更高效的算法
- SMO算法之所以高效，是因为其**每次只更新两个参数，而固定其他参数**，具体来说，考虑更新$$\alpha_i$$和$$\alpha_j$$，而固定其他参数，由于存在约束$$\sum_{i=1}^m{\alpha_iy_i} = 0$$，所以：

$$
\alpha_iy_i + \alpha_jy_j = c, \alpha_i \geq 0,\alpha_j \geq 0  \\ c = -\sum_{k \neq i, j}\alpha_ky_k
$$

c为一个已知的常数

- 则可以通过上式，消去$\alpha_j$，从而得到一个关于$\alpha_i$的单变量二次规划问题，仅有的约束是$\alpha_i \geq 0$，这样即可高效地更新$\alpha_i$，然后通过约束再得到更新后的$\alpha_j$

- 重复上述过程，每次只更新两个变量，直到收敛。但是每次按一定的规则选择两个变量进行更新：

> 因为选择的$\alpha_i, \alpha_j$只要有一个不满足KKT条件（一开始是随机初始化的），目标函数就会在迭代后增大，并且直观上来看，KKT条件违背的程度越大，则更新后获得的收益就越大。所以**SMO先选取先选取一个违背KKT条件程度最大的变量**，第二个变量应该选择使目标函数增长最快的变量，但是找出这个变量过于复杂，所以采用一个启发式：**使选取的两个变量所对应的样本之间的间隔最大**。直观解释为：差别大的两个变量的更新能给目标函数带来更大的增益

- 除了$\alpha$变量的更新，还需要确定偏移项b。对于任意支持向量$$(x_s, y_s)$$，都有$$y_sf(x_s) = 1$$，即：

$$
y_{s}\left(\sum_{i \in S} \alpha_{i} y_{i} \boldsymbol{x}_{i}^{\mathrm{T}} \boldsymbol{x}_{s}+b\right)=1
$$

其中S为所有支持向量的下标集。理论上，采用任意一个支持向量都可以得到b的值，但是SMO采用更鲁棒的做法：**使用所有支持向量求解的平均值：**
$$
b=\frac{1}{|S|} \sum_{s \in S}\left(y_{s}-\sum_{i \in S} \alpha_{i} y_{i} \boldsymbol{x}_{i}^{\mathrm{T}} \boldsymbol{x}_{s}\right)
$$





# 3 核函数

- 前面的讨论中，我们是假设训练样本是线性可分的，即存在一个划分超平面能将训练样本正确分类。然而在现实任务中，原始样本空间也许并不存在一个能正确划分两类样本的超平面。对于这样的问题，**可将样本从原始空间映射到一个更高维的特征空间，使得样本在这个特征空间中线性可分**，如下图：

<img src="https://zlkqzimg-1310374208.cos.ap-chengdu.myqcloud.com/image-20220923160040076.png" alt="image-20220923160040076" style="zoom:80%;" />

- 令$$\phi(x)$$表示将x映射后的特征向量，于是在特征空间中的超平面对应的模型为：

$$
f(x) = w^T\phi(x) + b
$$

- 然后像之前一样进行对偶问题的最优化，但是其中有一个内积项$$\phi(x_i)^T\phi(x_j)$$，这是样本$x_i$核$x_j$映射到特征空间后的内积。由于特征空间维度可能很高，甚至可能是无穷维，直接计算此内积项通常比较困难，所以提出了**核函数**：

$$
\mathcal{k}(x_i, x_j) = <\phi(x_i), \phi(x_j)> = \phi(x_i)^T\phi(x_j)
$$

**即$x_i$和$x_j$在特征空间的内积可以通过核函数在原始样本空间中的结果得出**

- 因此，在线性不可分问题中，核函数的选择成了支持向量机的最大变数，若选择了不合适的核函数，则意味着将样本映射到了一个不合适的特征空间，则极可能导致性能不佳。同时，**核函数需要满足以下这个必要条件**：

  ![26.png](https://zlkqzimg-1310374208.cos.ap-chengdu.myqcloud.com/5bc730ccc468c.png)

  由于核函数的构造十分困难，通常我们都是从一些常用的核函数中选择，下面列出了几种常用的核函数：

  ![27.png](https://zlkqzimg-1310374208.cos.ap-chengdu.myqcloud.com/5bc730ccc541a.png)





# 4 软间隔

- 前面讨论的情况是假定样本在样本空间或特征空间线性可分。然而在限时任务中往往很难确定合适的核函数，退一步讲，即使找到了核函数，也无法确定这个线性可分的结果是否是由于过拟合造成的。例如数据中有噪声的情况，噪声数据（outlier）本就偏离了正常位置，但是在前面的SVM模型中，我们要求所有的样本数据都必须满足约束，如果不要这些噪声数据还好，**当加入这些outlier后导致划分超平面被挤歪了**，如下图所示，对支持向量机的泛化性能造成很大的影响：

<img src="https://zlkqzimg-1310374208.cos.ap-chengdu.myqcloud.com/5bc730ccce68e.png" alt="28.png" style="zoom:67%;" />

可以看到如果不要outlier，能分出一个间隔更大的划分超平面

- 缓解这个问题的一个办法是允许SVM在一些样本上出错。前面所述的SVM要在所有样本上都划分正确，这成为**硬间隔（hard margin）**。而**软间隔则是允许某些样本不满足约束 $$y_{i}\left(\boldsymbol{w}^{\mathrm{T}} \boldsymbol{x}_{i}+b\right) \geqslant 1$$，但是不满足该约束的样本要尽可能少**，于是优化目标可以写为：

$$
\min _{\boldsymbol{w}, b} \frac{1}{2}\|\boldsymbol{w}\|^{2}+C \sum_{i=1}^{m} \ell_{0 / 1}\left(y_{i}\left(\boldsymbol{w}^{\mathrm{T}} \boldsymbol{x}_{i}+b\right)-1\right)
$$

其中C是一个常数，$$\ell_{0/1}(z)$$是0/1损失函数，即$$z < 0$$的时候取1，其他时候取0。**当C为无穷大的时候，则迫使所有样本满足约束，退化为硬间隔。当C为有限值的时候，允许一些样本不满足约束**

- 由于$$\ell_{0/1}$$非凸非连续，数学性质不太好，所以可以用以下三种替代损失函数：

> - hinge损失：$$\ell_{\text {hinge }}(z)=\max (0,1-z)$$
> - 指数损失（exponential loss）：$$\ell_{\exp }(z)=\exp (-z) $$
> - 对率损失（logistic loss）：$$\ell_{\log }(z)=\log (1+\exp (-z))$$

- 常用hinge损失进行替代，然后将连加中的每一项换为松弛变量（slack variables）$\xi_{i} \ge 0$，则优化目标重写为：

$$
\min _{\boldsymbol{w}, b, \xi_{i}} \frac{1}{2}\|\boldsymbol{w}\|^{2}+C \sum_{i=1}^{m} \xi_{i}
$$

每个变量都对应一个松弛变量，代表**样本不满足约束的程度**

- 上述问题仍是一个二次规划问题，按照和前面一样的方法进行求解，先写出拉格朗日函数：

$$
\begin{aligned}
L(\boldsymbol{w}, b, \boldsymbol{\alpha}, \boldsymbol{\xi}, \boldsymbol{\mu})=& \frac{1}{2}\|\boldsymbol{w}\|^{2}+C \sum_{i=1}^{m} \xi_{i} \\
&+\sum_{i=1}^{m} \alpha_{i}\left(1-\xi_{i}-y_{i}\left(\boldsymbol{w}^{\mathrm{T}} \boldsymbol{x}_{i}+b\right)\right)-\sum_{i=1}^{m} \mu_{i} \xi_{i}
\end{aligned}
$$

其中$\alpha_i \ge 0$、$\mu_i \ge 0$是拉格朗日乘子，据此求解即可





