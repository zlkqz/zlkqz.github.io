---
title: SimCSE总结
math: true
date: 2022-10-21
---





- 作者提出了一种对比学习的方法，分为**有监督和无监督两种**。其中**只用dropout作为噪音，也可以当作一种数据增强，可以改善语义空间，提升其同向异性，使向量空间更为均匀，并且在有监督方法时，还能对齐正样本**



# 1 评估标准

- 本篇文章的目的是改善embedding，并且作者在后面的实验发现，进行了SimCSE后有部分下游任务上的表现甚至出现了下降，但是这并不影响SimCSE的作用。**句子嵌入的主要目标是对语义相似的句子进行聚类，所以为了更加综合的评估实验结果，肯定不能使用某个下游任务的实验结果**，作者采用了另一篇论文中一种评估embedding质量的方法：**采用语义相关的正样本之间的对齐（alignment）和整个表示空间的一致性（uniformity）来衡量学习嵌入的质量**

> We takes **alignment** between semantically-related positive pairs and **uniformity** of the whole representation space to measure the quality of learned embeddings.  

- 总的来说，对比学习所做的任务就是：**拉近正样本的距离，剩余的随机样本应该均匀分布在一个超平面上（也就是减少其各向异性）**，所以对比学习的任务就变为了降低以下两个指标：

$$
\ell_{\text {align }} \triangleq \underset{\left(x, x^{+}\right) \sim p_{\text {pos }}}{\mathbb{E}}\left\|f(x)-f\left(x^{+}\right)\right\|^{2}, \\
\ell_{\text {uniform }} \triangleq log \underset{\left(x, y\right) \sim p_{\text {data }}}{\mathbb{E}}e^{-2\left\|f(x)-f\left(y\right)\right\|^{2}}
$$

其中$$p_{pos}$$为正样本对，$$p_{data}$$为所有数据对，$$f(x)$$为输入$$x$$经过encoder的输出

- 并且作者还发现无监督的SimCSE能够向量空间的均匀性，并且并不会降低正样本之间的对齐。然后对于有监督，作者指出NLI任务最为适合训练出好的sentence embedding，并且有监督能够进一步提升正样本之间的对齐

- 本文还多次使用了STS-B数据集，这是一个五分类任务的数据集，旨在判定两个句子的相关程度，分为了5个等级，并且得分采用斯皮尔曼等级相关系数





# 2 无监督SimCSE

### 2.1 基本方法

- 方法非常简单，就是将同一个输入，分别经过两次encoder，encoder中的dropout**（dropout率仍为默认的0.1）**作为一种微小的数据增强，会使得两次的输出有些许不同。这两次的输出，就作为一对正样本，然后使用以下loss：

$$
\ell_{i}=-\log \frac{e^{\operatorname{sim}\left(\mathbf{h}_{i}^{z_{i}}, \mathbf{h}_{i}^{z_{i}^{\prime}}\right) / \tau}}{\sum_{j=1}^{N} e^{\operatorname{sim}\left(\mathbf{h}_{i}^{z_{i}}, \mathbf{h}_{j}^{z_{j}^{\prime}}\right) / \tau}},
$$

其中$$h_i^z=f_{\theta}(x_i, z)$$为输入$$x_i$$经过$$\theta$$的encoder进行编码得到的结果，其中的z代表不同的dropout mask，每次的dropout mask都不同。N为batch size，所以该loss是每个batch内的交叉熵。$$\tau$$为温度超参。sim()使用的是余弦距离

- **并且在微调时选择更新所有参数**



### 2.2 Dropout和其他数据增强方式的对比

- 本文是将dropout作为一种微小的数据增强方式，所以作者也将其他数据增强方式同其对比了一下，本实验采用lr=3e-5，N=64，结果如下：

<img src="https://zlkqzimg-1310374208.cos.ap-chengdu.myqcloud.com/image-20220810133558725.png" alt="image-20220810133558725" style="zoom:75%;" />

- **发现其他数据增强方式都没有SimCSE效果好（可能是其他方法噪音太大了）**



### 2.3 采用一个OR两个Encoder

- 由于之前有些论文是使用的两个不同的encoder，所以作者也就采用一个还是两个encoder的问题进行了对比试验，结果如下：

<img src="https://zlkqzimg-1310374208.cos.ap-chengdu.myqcloud.com/image-20220810140539627.png" alt="image-20220810140539627" style="zoom:80%;" />

图中的next sentence为输入原句子和该句子的下一句。Delete on word同2.2中图一样，输入原句子和删除一个词的原句

- 通过实验发现，**只用一个encoder比两个要好**



### 2.4 采用多少Dropout率

- dropout是SimCSE中重要的一环，所以作者对该超参进行了实验，结果如下：

<img src="https://zlkqzimg-1310374208.cos.ap-chengdu.myqcloud.com/image-20220810141017109.png" alt="image-20220810141017109" style="zoom:80%;" />

图中的fixed 0.1为0.1的dropout率，但是对正样本对中的两个样本使用相同的dropout mask，就是两个输出都长一样（有用才有怪了）

- **通过实验发现，还是原先默认的0.1最好用**



### 2.5 alignment and uniformity

- 前面说过，最综合的评估标准是检测结果向量空间的alignment和uniformity，作者对几种方法进行了评估，并给出了可视化的结果：

<img src="https://zlkqzimg-1310374208.cos.ap-chengdu.myqcloud.com/image-20220810142955971.png" alt="image-20220810142955971" style="zoom:80%;" />

其中箭头所指方向是训练进行的方向，横轴和竖轴都是越小越好

- 通过实验发现，**所有的方法都能有效的提升uniformity**，但是前两种方法会降低正样本之间的alignment，而**无监督SimCSE的alignment则稳定不变**，delete one word可以稍微增加alignment，但是总体表现还是低于无监督SimCSE





# 3 有监督SimCSE

- 无监督的SimCSE可以提升uniformity，但是alignment不会有改善。而之后作者引入了有监督的数据，**利用其提供更好的训练信号，以提升alignment**



### 3.1 使用哪种有监督数据

- 先简要介绍一下SNLI和MNLI数据集，都是NLI任务下的数据集，是一个三分类，每次输入两个文本，模型预测两者的相似度，然后进行分类：**entailment（相关）、neutral（无关）、contradiction（矛盾）**，举个栗子：

<img src="https://zlkqzimg-1310374208.cos.ap-chengdu.myqcloud.com/image-20220810152937757.png" alt="image-20220810152937757" style="zoom:67%;" />

- 作者探究了使用哪种有监督的数据集，能更有效地提升SimCSE的性能，结果如下：

<img src="https://zlkqzimg-1310374208.cos.ap-chengdu.myqcloud.com/image-20220810152252447.png" alt="image-20220810152252447" style="zoom:67%;" />

图中sample指在数据集中采样了134k的正样本对，full指使用整个数据集。最后两行是使用NLI任务中的entailment对做正样本，contradiction对做负样本（把neutral对丢了）

- 作者发现**使用NLI任务的数据集效果最显著，并且加上hard negative能进一步提升表现**
- 并且作者还又尝试使用两个encoder，但是表现下降了



### 3.2 基本方法

- 相比于无监督，有监督将每个样本对$$(x_i, x_i^+)$$拓展为了三元组$$(x_i,x_i^+,x_i^-)$$，其中$$x_i^+$$和$$x_i^-$$分别为$$x_i$$的entailment样本和contradiction样本，然后采用以下loss：

$$
-\log \frac{e^{\operatorname{sim}\left(\mathbf{h}_{i}, \mathbf{h}_{i}^{+}\right) / \tau}}{\sum_{j=1}^{N}\left(e^{\operatorname{sim}\left(\mathbf{h}_{i}, \mathbf{h}_{j}^{+}\right) / \tau}+e^{\operatorname{sim}\left(\mathbf{h}_{i}, \mathbf{h}_{j}^{-}\right) / \tau}\right)}
$$

- 但是从**直观上**来讲，区分难负例（矛盾文本）和Batch内其他负例可能是有益的，所以将有监督学习SimCSE的训练目标变成：

$$
-\log \frac{e^{\operatorname{sim}\left(\mathbf{h}_{i}, \mathbf{h}_{i}^{+}\right) / \tau}}{\sum_{j=1}^{N}\left(e^{\operatorname{sim}\left(\mathbf{h}_{i}, \mathbf{h}_{j}^{+}\right) / \tau}+\alpha^{\mathbb{1}_{i}^{j}} e^{\operatorname{sim}\left(\mathbf{h}_{i}, \mathbf{h}_{j}^{-}\right) / \tau}\right)}
$$

其中$$1_i^j \in \{0, 1\}$$仅当$$i=j$$时为1

- 作者对不同的$$\alpha$$进行了实验，结果如下：

<img src="https://zlkqzimg-1310374208.cos.ap-chengdu.myqcloud.com/image-20220810165606726.png" alt="image-20220810165606726" style="zoom:80%;" />

其中N/A为不使用hard negative

- **由上表可以得到$$\alpha=1$$最合适（其实就是又退化回去了，没啥用......），并且将Neural的样本一起作为负例并不能提升表现**





# 4 各向异性问题

- 最近的研究发现了语言表征中的各向异性问题，**即训练后的embeddings仅占据在向量空间中狭窄的部分，严重限制了向量的表现力**。缓解这个问题的一个简单方法是**后处理**，可以**消除主要的主成分或将embeddings映射到各向同性分布**。另一种常见的解决方案是在**训练过程中添加正则项**。 而对比学习的优化目标可以改善缓解各向异性问题，当负例数趋近于无穷大时，对比学习目标的渐近表示为:

$$
-\frac{1}{\tau} \underset{\left(x_{i}, x_{i}^{+}\right) \sim p_{p o s}}{E}\left[f(x)^{T} f\left(x^{+}\right)\right]+\underset{x \sim p_{\text {data }}}{E}\left[\log \underset{x^{-} \sim p_{\text {data }}}{E}\left[e^{f(x)^{T} f\left(x^{-}\right) / \tau}\right]\right]
$$

其中，**第一项使正例之间更相似，第二项使将负例之间分开。**而第二项**在优化过程中，会压平向量空间的奇异谱，因此对比学习有望缓解表征退化问题，提高句向量表征的均匀性**

- 并且作者还针对不同的模型、不同的后处理方法、不同的数据扩充方法等，通过alignment和uniformity进行了实验：

<img src="https://zlkqzimg-1310374208.cos.ap-chengdu.myqcloud.com/image-20220810192852943.png" alt="image-20220810192852943" style="zoom:80%;" />

图中括号中的是再STS任务上的得分

- 通过上图，可以得出以下结论：

> 1. 虽然预训练embeddings具有良好的对齐性，但其均匀性较差
> 2. 后处理方法，如BERT-flow和BERT-whitening，大大改善均匀性，但也使其对齐性变差
> 3. 无监督SimCSE有效地提高了预训练embeddings的均匀性，同时保持了良好的对齐性
> 4. 有监督SimCSE，可以进一步提高对齐性





# 5 对比试验

### 5.1 STS任务上的对比

- 作者先在7个STS任务上进行了对比实验，结果如下：

<img src="https://zlkqzimg-1310374208.cos.ap-chengdu.myqcloud.com/image-20220810170503226.png" alt="image-20220810170503226" style="zoom:70%;" />

- 可以发现，无监督和有监督的SimCSE均取得了SOTA的效果，并且同时适用于BERT和RoBERTa



### 5.2 Pooling方式

- 在实验中，是采用[CLS]的表征进行分类的，但是有其他文章表示使用embedding的平均能提升表现。并且如果采用[CLS]，原始的BERT在其之后添加了一个额外的MLP层，本文对MLP同样有三种pooling方式：(1)、保留MLP层；(2)、丢弃MLP层；(3)、训练时采用MLP层，测试时丢弃。实验结果如下：

<img src="https://zlkqzimg-1310374208.cos.ap-chengdu.myqcloud.com/image-20220810172926336.png" alt="image-20220810172926336" style="zoom:80%;" />

- 从结果中得知：**无监督在train中使用MLP，test中丢弃MLP表现最好；有监督不同的pooling方法不是差别不是很大**
- **作者选择在无监督中使用MLP(train)，而在有监督中使用with MLP**



### 5.3 召回任务的结果

- 作者还使用$$SBERT_{base}$$和$$SimCSE-BERT_{base}$$进行了一个小规模的召回实验，给定query，找出相似的句子（基于余弦相似度），结果如下：

![image-20220810185441782](https://zlkqzimg-1310374208.cos.ap-chengdu.myqcloud.com/image-20220810185441782.png)

- 结果是，SimCSE找出的句子质量更高



### 5.4 温度超参和相似度函数的选择

- 作者尝试使用了不同的$$\tau$$超参，并且尝试用点积代替余弦相似度，结果如下：

![image-20220810194557958](https://zlkqzimg-1310374208.cos.ap-chengdu.myqcloud.com/image-20220810194557958.png)

N\A表示点积代替余弦相似度

- 发现**使用余弦相似度更合适，并且$$\tau=0.05$$表现最好**





# 6 下游任务上的表现

- 作者还在各种下游任务上进行了对比，并且加上了MLM任务（BERT中的MLM任务），**避免模型彻底的忘记token-level的知识，并发现加上MLM后可以在除STS任务外的其他下游任务上取得提升**，加上MLM后，训练目标由原本的$$\ell$$变成了$$\ell + \lambda \cdot \ell ^{MLM}$$



### 6.1 MLM的对比

- 作者对比了在STS任务和其他下游任务上，加与不加MLM的结果对比，以及$$\lambda$$超参的选择：

![image-20220810195101507](https://zlkqzimg-1310374208.cos.ap-chengdu.myqcloud.com/image-20220810195101507.png)

- 结果说明，**添加token-level对于其他大多数下游任务都有提升，并且$$\lambda=0.1$$最为合适，但是这会带来STS任务表现的下降**



### 6.2 下游任务的对比

- 最后作者给出了在各种模型、训练策略、处理方式等因素不同时，在各种下游任务上的表现：

<img src="https://zlkqzimg-1310374208.cos.ap-chengdu.myqcloud.com/image-20220810195709648.png" alt="image-20220810195709648" style="zoom:80%;" />

- 可以发现在迁移任务上该方法并没有做到最好，不过这也证明了作者的说法，句子级别的目标可能并不会有益于下游任务的训练，训练好的句子向量表示模型也并不是为了更好的适应下游任务，但是SimCSE也在许多任务上做到了SOTA，特别是带MLM的时候

