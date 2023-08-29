---
title: LoRA && QLoRA
math: ture
date: 2023-08-18
---



# 1 前置假设

- LoRA的灵感来自于另一篇论文：Intrinsic Dimensionality Explains the Effectiveness of Language Model Fine-Tuning
- 这篇论文认为，**在参数更新时，不需要对参数的整个向量空间进行更新，只需要随机映射到他的一个子空间（一般都是直接指列空间的子空间），在这个子空间上参数更新，就能达到十分接近的效果，能达到这个要求的最小子空间维度即为“intrinsic dimension”**
- 设原参数为$$W \in \mathbb{R}^{D \times m}$$，其列空间为$$\theta^m = [\theta_0, ..., \theta_{m-1}]$$，那么参数更新公式则变为了：

$$
\theta^m = \theta^m + P(\theta^n)
$$

其中$$\theta^n \in \mathbb{R}^{D \times n}$$表示映射后的子空间，P表示$$\mathbb{R}^n \rightarrow \mathbb{R}^m$$的随机映射，用矩阵表示为右乘$$M \in \mathbb{R}^{n \times m}$$。**子空间维度n即为“intrinsic dimension”**

- 随机映射P有多种，包括随机线性映射、随机稀疏线性映射、Fastfood Transform等。并且$$\theta^n$$是初始化为全0以保持开始时和预训练权重一致

- 这篇文章还有其他结论：

> 1. 预训练可以降低intrinsic dimension
> 2. 大模型通常拥有更小的intrinsic dimension





# 2 LoRA

### 2.1 LoRA定义

- LoRA认为**权重的更新量同样拥有一个intrinsic dimension（记为r），所以将权重更新量$$\Delta W$$分解为两个矩阵$$\Delta W = BA$$**

- 在计算时，将原预训练权重的activation加上这个BA的activation，得到新的activation：

$$
h=W_{0} x+\Delta W x=W_{0} x+B A x
$$

其中$$W_0 \in \mathbb{R}^{d \times k}$$，$$B \in \mathbb{R}^{d \times r}$$表示映射后的子空间参数矩阵，$$A \in \mathbb{R}^{r \times k}$$表示两个空间的映射矩阵，并且$$r \ll \min (d, k)$$

- 在参数初始化时是将B设为全0，A服从$$\mathcal{N}\left(0, \sigma^{2}\right)$$：

<img src="https://zlkqzimg-1310374208.cos.ap-chengdu.myqcloud.com/image-20230808143823489.png" alt="image-20230808143823489" style="zoom:60%;" />



### 2.2 LoRA的优点

- **在训练时，是将原参数冻结，然后只更新A、B矩阵**，由于$$r \ll \min (d, k)$$，所以用LoRA微调，需要更新的参数很少，并且尽管可更新参数变小了，但是效果却不差
- LoRA没有像Adapter那样的额外推理时间消耗，**在推理部署时，是直接将BA的结果加到原参数$$W_0$$上，再部署上去**



### 2.3 对比实验

- LoRA可以应用的地方有5个，分别是Attention Layer里的$$W_q, W_k, W_v,W_o$$矩阵，以及MLP中的权重矩阵。**本次实验并未对MLP的权重矩阵进行LoRA优化**
- **并且在大部分对比实验中，都是仅对$$W_q, W_v$$进行$$r=4$$的优化**

##### 2.3.1 和其他微调方法的对比

- RoBERTa结果：

<img src="https://zlkqzimg-1310374208.cos.ap-chengdu.myqcloud.com/image-20230808172556109.png" alt="image-20230808172556109" style="zoom:40%;" />

- GPT-2结果：

<img src="https://zlkqzimg-1310374208.cos.ap-chengdu.myqcloud.com/image-20230808172638176.png" alt="image-20230808172638176" style="zoom:40%;" />

- GPT-3结果：

<img src="https://zlkqzimg-1310374208.cos.ap-chengdu.myqcloud.com/image-20230808172719697.png" alt="image-20230808172719697" style="zoom:40%;" />

- 此外，还对比了当可训练参数逐渐增多时，各种微调方法的结果变化（GPT-3上）：

<img src="https://zlkqzimg-1310374208.cos.ap-chengdu.myqcloud.com/image-20230808172934424.png" alt="image-20230808172934424" style="zoom:50%;" />



##### 2.3.2 对哪些矩阵采用LoRA

- 作者在相同的可训练参数量的情况下，实验了不同的矩阵和不同的r：

<img src="https://zlkqzimg-1310374208.cos.ap-chengdu.myqcloud.com/image-20230808173756918.png" alt="image-20230808173756918" style="zoom:45%;" />

- 可以发现，**同时对$$W_q, W_v$$进行优化，取得的效果最好**
- 并且对比单用$$W_q$$或者单用$$W_k$$的情况，发现**对于每个矩阵$$r=4$$已经能获取足够的特征，这时将LoRA应用更多的矩阵比单纯提升$$r$$更有用**



##### 2.3.3 最佳r

- 作者实验了应用不同的矩阵和不同的r：

<img src="https://zlkqzimg-1310374208.cos.ap-chengdu.myqcloud.com/image-20230808174526241.png" alt="image-20230808174526241" style="zoom:45%;" />

- 表明在很小的r上，LoRA已经能够取得很好的结果



### 2.4 Empirical Analysis

##### 2.4.1 如何对比子空间的相似度

- 给定两个矩阵$$A, B$$，而$$U_A^i \in \mathbb{R}^{d \times i}$$表示A的左或右奇异矩阵的前$$i$$列组成的矩阵，$$U_B^j \in \mathbb{R}^{d \times j}$$同理，则这两个奇异矩阵的子空间的相似度可以定义为：

$$
\phi(A, B, i, j)=\psi\left(U_{A}^{i}, U_{B}^{j}\right)=\frac{\left\|U_{A}^{i \top} U_{B}\right\|_{F}^{2}}{\min \{i, j\}}
$$

- 具体到LoRA上面，$$\Delta W = BA$$，我们对子空间的分析可以采用**B的左奇异矩阵**或者**A的右奇异矩阵**，论文是采用后者来分析的



##### 2.4.2 不同r所产生的子空间

- 作者采用相同的预训练权重，用$$r=8$$和$$r=64$$分别微调了一遍，并得到了两个A矩阵$$A_{r=8}, A_{r=64}$$，以及他们的右奇异矩阵$$U_{A_{r=8}}, U_{A_{r=64}}$$，并计算他们不同子空间的相似度：

$$
\phi\left(A_{r=8}, A_{r=64}, i, j\right)=\frac{\left\|U_{A_{r=8}}^{i \top} U_{A_{r=64}}^{j}\right\|_{F}^{2}}{\min (i, j)} \in[0,1]
$$

- 得到以下结果：

<img src="https://zlkqzimg-1310374208.cos.ap-chengdu.myqcloud.com/image-20230810153943129.png" alt="image-20230810153943129" style="zoom:50%;" />

右边两图对应左边两图的左下角空白处

- 分析结论：**在$$i, j$$较小时，是由top的几个奇异向量组成的子空间，在不同的r下，这个子空间的相似度是非常高的（特别是在$$i=1$$或$$j=1$$时，子空间的相似度甚至大于0.5）。说明top的几个奇异向量组成的子空间才是最主要的，这也解释了为什么GPT-3用$$r=1$$都能取得很不错的结果**



##### 2.4.3 $$\Delta W_q$$和$$\Delta W_v$$的对比

- 作者采用了$$r=64$$，但是不用预训练权重，而是两次不同的随机初始化，来对模型进行训练，得到了两个不同的A矩阵$$A_{r=64}, A'_{r=64}$$，得到结果：

<img src="https://zlkqzimg-1310374208.cos.ap-chengdu.myqcloud.com/image-20230810161204970.png" alt="image-20230810161204970" style="zoom:50%;" />

- 分析结论：**在不同的初始参数下，$$\Delta W_q$$共享重合的子空间（即相似度高的奇异矩阵子空间）的维度比$$\Delta W_v$$更高，说明$$\Delta W_q$$具有更高的intrinsic dimension，表明$$\Delta W_q$$学得的下游"task specific"信息更多**



##### 2.4.4 $$W$$和$$\Delta W$$的关系

- 作者还对比了$$W_q$$以及其$$A_{r=k}$$的子空间相似度：

<img src="https://zlkqzimg-1310374208.cos.ap-chengdu.myqcloud.com/image-20230810162338348.png" alt="image-20230810162338348" style="zoom:50%;" />

- 分析结论：**在$$i = 400$$周围，他们的子空间是几乎不重叠的，但是随着$$i$$的增加，子空间相似度居然又上来了。说明$$\Delta W$$所包含的奇异向量是$$W$$的奇异空间中很靠后的奇异向量。说明LoRA潜在地强调了一些重要特征，而这个特征是包含在预训练中，但是并没有在预训练中强调出来的task-specific特征**

> This suggests that the low-rank adaptation matrix potentially amplifies the important features for specific downstream tasks that were learned but not emphasized in the general pre-training model  





# 3 QLoRA

### 3.1 量化技术

- 最常见的量化方法是：

$$
\begin{gather}
Q = \frac{R}{S} + Z \\
R =(Q - Z) * S
\end{gather}
$$

其中R表示量化前的值，Q表示量化后的值，$$S = \frac{R_{\max} - R_{\min}}{Q_{\max} -Q_{min}}$$表示缩放系数，$$Z = Q_{\max} - R_{\max}/S$$表示偏移量

- 而在QLoRA中运用了更简化一点的形式：

$$
\begin{gather}
\mathbf{X}^{\mathrm{Int8}}=\operatorname{round}\left(\frac{127}{\operatorname{absmax}\left(\mathbf{X}^{\mathrm{FP} 32}\right)} \mathbf{X}^{\mathrm{FP} 32}\right)=\operatorname{round}\left(c^{\mathrm{FP} 32} \cdot \mathbf{X}^{\mathrm{FP} 32}\right) \\
\operatorname{dequant}\left(c^{\mathrm{FP} 32}, \mathbf{X}^{\mathrm{Int} 8}\right)=\frac{\mathbf{X}^{\mathrm{Int} 8}}{c^{\mathrm{FP} 32}}=\mathbf{X}^{\mathrm{FP} 32}
\end{gather}
$$

- 运用该方法主要是可以少存储一个偏移量
- 但是量化方法有一个通病：**我们是希望量化后的每一个bit的利用率都尽可能高，即映射到每个bit的概率是差不多的，这样信息损失才更小。但是如果有很大的离群值，那么bit利用率会很低**
- 解决方法：**采用分块量化，即将一个Tensor分为几个小块，分别量化和分别存储量化系数$$c^{\mathrm{FP} 32}$$**



### 3.2 QLoRA实现

##### 3.2.1 4-bit NormalFloat Quantization

- QLoRA是采用分位数量化，是将权重量化至4bit，并且量化的block_size为64，步骤如下：

1. 采集$$N(0,1)$$的18分位点（有17个分位点），然后通过下式计算：

$$
q_{i}=\frac{1}{2}\left(Q_{X}\left(\frac{i}{2^{k}+1}\right)+Q_{X}\left(\frac{i+1}{2^{k}+1}\right)\right)
$$

得到16个$$q_i$$，然后将这16个$$q_i$$除以$$absmax(q_i)$$，将其映射到$$[-1,1]$$，然后保存下来

2. 然后再将原权重除以量化常数$$c^{\mathrm{FP} 32}$$映射到$$[-1,1]$$**（注意这个$$c^{\mathrm{FP} 32}$$是要保存下来的，去量化的时候用）**，和上一步保存的$$q_i$$对比，把原参数转为最近的$$q_i$$的index
3. 去量化时再通过index查保存下来的$$q_i$$，再乘上之前除的即$$c^{\mathrm{FP} 32}$$可

- 上述方法存在问题：**通过第一步得到的$$q_i$$中并没有0，所以在去量化后不会得到0，但是0在计算中是一个非常重要的值，所以去量化后必须得到准确的0**

- 解决方法：**再加一个分位点，使用19分位点（有18个分位点），在17个得到$$q_i$$后（映射前的$$q_i$$），再将前8个映射到$$[-1,0]$$，后9个映射到$$[0, 1]$$，再将两部分合并（会丢弃一个重合的0），从而得到16个映射后的$$q_i$$**

- 论文给出了通过这种方法所得到的$$q_i$$，直接保存下来使用即可，不需要现算：

<img src="https://zlkqzimg-1310374208.cos.ap-chengdu.myqcloud.com/image-20230829160539803.png" alt="image-20230829160539803" style="zoom:40%;" />



##### 3.2.2 双量化

- 上面说过，对于每个block（block_size=64），需要保存一个$$c^{\mathrm{FP} 32}$$，而这会造成$$32/64=0.5$$bits/parameter的内存增加
- 而双量化则是对这个$$c^{\mathrm{FP} 32}$$再做一次量化，将其映射到8bit，由于这个absmax很少会有离群值，所以block_size可以大些，第二次量化的block_size为256，这样就把内存增加降低至$$8/64 + 32/(64 \times 256)=0.127$$bits/parameter

- 并且由于$$c^{\mathrm{FP} 32}$$都为正，所以直接量化只会用到一半的bit，所以是先减去均值再量化。在去量化的时候，乘上$$c^{\mathrm{FP} 32}$$之后再算一波均值，然后再加上这个均值

> - 个人觉得这个双量化没啥必要哈，节省的内存也很有限说实话
> - 另外QLoRA还用了Paged Optimizers优化了一些CPU、GPU内存调用的问题



##### 3.2.3 具体实现

- 首先，用上述方法将模型权重量化到4bit得到$$W^{\mathrm{NF4}}$$，再放入显存。训练是使用bf16精度，每次的计算步骤为：

$$
\begin{gather}
\mathbf{Y}^{\mathrm{BF} 16}=\mathbf{X}^{\mathrm{BF} 16} \text { doubleDequant }\left(c_{1}^{\mathrm{FP} 32}, c_{2}^{8\mathrm{bit}}, \mathbf{W}^{\mathrm{NF} 4}\right)+\mathbf{X}^{\mathrm{BF} 16} \mathbf{L}_{1}^{\mathrm{BF} 16} \mathbf{L}_{2}^{\mathrm{BF} 16} \\
\operatorname{doubleDequant}\left(c_{1}^{\mathrm{FP} 32}, c_{2}^{8 \text {bit }}, \mathbf{W}^{4 \text {bit }}\right)=\operatorname{dequant}\left(\operatorname{dequant}\left(c_{1}^{\mathrm{FP} 32}, c_{2}^{8\mathrm{bit}}\right), \mathbf{W}^{4 \mathrm{bit}}\right)=\mathbf{W}^{\mathrm{BF} 16}
\end{gather}
$$



### 3.3 对比实验

##### 3.3.1 LoRA超参选择

- LoRA论文中采用的只微调query和value矩阵，QLoRA还对比了其他超参，如全参数微调（Alpaca的训练方法）、只调FFN、对所有Attention Layer微调，实验结果：

<img src="https://zlkqzimg-1310374208.cos.ap-chengdu.myqcloud.com/image-20230829174543969.png" alt="image-20230829174543969" style="zoom:33%;" />

- 作者发现，**原LoRA参数在QLoRA中并不能获得和全参数微调持平的结果，但是对所有Attention Layer微调可以，并且后面还有个实验说明了$$r$$的选择其实不是很重要**

- 另外，对比右边两列，发现Alpaca的超参选择的不好，所以自己又调了一版，结果更好



##### 3.3.2 NF4 && 双量化

- 作者对比了NF4和其他4-bit精度，并且探究了双量化对效果是否有影响，实验结果：

<img src="https://zlkqzimg-1310374208.cos.ap-chengdu.myqcloud.com/image-20230829175203934.png" alt="image-20230829175203934" style="zoom:33%;" />

<img src="https://zlkqzimg-1310374208.cos.ap-chengdu.myqcloud.com/image-20230829175357381.png" alt="image-20230829175357381" style="zoom:40%;" />



### 3.4 Chat-Bot Evaluation

##### 3.4.1 训练配置

- 用**OASST1、HH-RLHF、Alpaca 、self-instruct、unnatural-instructions、FLAN v2、Chip2、Longform**等几个数据集微调LLaMA，**并没有进行RLHF**
- 测评采用**MMLU、对比ChatGPT、Elo Rating**，MMLU采用5-shot测试，后两者是基于Vicuna prompts和OASST1的测试集（称为Vicuna benchmark）进行测评的，并且混合采用了人类和GPT-4进行测评
- 对比ChatGPT时，**对于同一prompt，测评模型和ChatGPT生成的responses，分别打分（满分10分）。**再计算该模型能达到ChatGPT性能的百分之几，100%为刚好打平
- 计算Elo Rating时，**对于同一prompt，不同模型responses，进行三分类任务，选哪个更好或打平。**再通过多次对比结果，计算Elo Rating

- 在采用GPT-4进行测评时，**发现GPT-4会对在prompt中排靠前的answer有更多偏好，所以每次用GPT-4测评两次，依次将不同answer至于靠前位置，然后取平均分**

- 其中Guanaco是用QLoRA+OASST1数据集的变体微调的模型



##### 3.4.2 实验结果

- **不同数据集的微调结果：**

<img src="https://zlkqzimg-1310374208.cos.ap-chengdu.myqcloud.com/image-20230829191239069.png" alt="image-20230829191239069" style="zoom:40%;" />

- **Competition with ChatGPT：**

<img src="https://zlkqzimg-1310374208.cos.ap-chengdu.myqcloud.com/image-20230829191321959.png" alt="image-20230829191321735" style="zoom:35%;" />

- **Elo rating排行榜：**

<img src="https://zlkqzimg-1310374208.cos.ap-chengdu.myqcloud.com/image-20230829191406666.png" alt="image-20230829191406666" style="zoom: 37%;" />



##### 3.4.3 结果分析

- Guanaco能达到和ChatGPT差不多的水平，**表明QLoRA在chat-bot表现上仍然很好**
- **数据集和下游任务的匹配度才是最重要的**，比如FLAN v2在MMLU上表现很好，但是在Vicuna benchmark上表现很差
- 并且还说明了MMLU并不能很全面地反映chat-bot的能力（个人理解：MMLU反映的是在多学科、多领域的知识能力以及知识覆盖面，而其他chat-bot能力反应的不是特别好）