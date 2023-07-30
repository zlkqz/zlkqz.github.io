---
title: GPT
math: ture
date: 2023-3-10
---



# 1 GPT v1

- GPT采用无监督预训练+下游任务微调的方法

### 1.1 模型结构
- 采用12个堆叠的Transformer的Decoder块（去除了和encoder连接的那个Multi-Head）：

  <img src="https://zlkqzimg-1310374208.cos.ap-chengdu.myqcloud.com/IMG_1.png" alt="IMG_1" style="zoom:40%;" />



### 1.2 模型训练目标

##### 1.2.1 无监督预训练目标
- 无监督预训练采用的是LM（语言模型）的训练方法，采用n元语法：
$$
L_{1}(\mathcal{U})=\sum_{i} \log P\left(u_{i} \mid u_{i-k}, \ldots, u_{i-1} ; \Theta\right)
$$
其中k即n元语法中的n，具**体实现中k是取最大，即表示使用前面的所有词（个人觉得他这里说的有点歧义）**，$$\Theta$$是模型参数
- 具体到模型实现上， 类似于word2vec的实现方法，当要预测当前时间步的词u时，采用前面所有的词$$U = (u_{-k}, ..., u_{-1})$$来进行预测：
$$
\begin{aligned}
h_{0} & =U W_{e}+W_{p} \\
h_{l} & =\text { transformer_block }\left(h_{l-1}\right) \forall i \in[1, n] \\
P(u) & =\operatorname{softmax}\left(h_{n} W_{e}^{T}\right)
\end{aligned}
$$
其中$$W_e \in (vocab\_size, embedding\_dim)$$是embedding矩阵，$$W_p \in (seq\_len, embedding\_dim)$$是**学习到的**位置编码，n表示Transformer层数。**注意最后还是乘的$$W_e$$表示使用了Weight Tying**。具体实现是和Transformer一样的

##### 1.2.2 有监督微调
- 有监督任务一般都是在最后接一个全连接，训练目标是：
$$
L_{2}(\mathcal{C})=\sum_{(x, y)} \log P\left(y \mid x^{1}, \ldots, x^{m}\right)
$$
其中x是输入，y是label
- 在微调的时候，作者还加入了LM无监督任务作为额外目标，那么微调时的训练目标变为：
$$
L_{3}(\mathcal{C})=L_{2}(\mathcal{C})+\lambda * L_{1}(\mathcal{C})
$$
其中$$\lambda$$表示权重
- **这么做的优点是：**可以增加模型泛化能力和收敛速度，后面作者还对此做了消融实验



### 1.3 微调具体实现方法

- GPT针对不同类型的下游任务，其做法是不同的。尤其是**由于在预训练时，是在连续通顺文本上训练的，所以在下游任务上有多个输入时，句子之间的相对顺序尤为重要**
- 最初的输入还要加三个特殊token：起始token（\<s\>）、分隔token（$）、结束token（\<e\>）
- **方法汇总：**

<img src="https://zlkqzimg-1310374208.cos.ap-chengdu.myqcloud.com/IMG_2.png" alt="IMG_2" style="zoom: 67%;" />

- 在做Textual entailment任务时，由于前提p和假设h是有前后文关系的，所以直接p在前h在后即可，中间用$做分隔
- 在做Similarity任务时，因为没有明确前后文关系，所以将两种排列顺序分别通过模型，最后将输出结果按元素相加，再喂入mlp
- 在做QA或尝试推理这类多选择任务时，上下文在前，选择在后，如给定背景上下文z、问题q、回答集$$\{a_k\}$$，那么分别构造$$[z;q;\$;a_k]$$作为输入。最后将结果通过softmax映射为概率



### 1.4 模型训练

##### 1.4.1 无监督预训练
- 采用Adam算法，并且加了warm up，最大学习率为2.5e-4
- epoch = 100，batch size = 64
- 采用$$N(0, 0.02)$$进行参数初始化，由于含有Layer Norm，所以初始化不需要太关注
- 激活函数采用GELU

##### 1.4.2 有监督微调
- 在mlp中也加入了dropout
- learning rate = 6.25e-5，batch size = 32, epochs = 3
- 采用线性学习率衰减，在0.2%的训练中使用了warm up，超参$$\lambda = 0.5$$



### 1.5 下游任务表现
- **NLI任务：**

<img src="https://zlkqzimg-1310374208.cos.ap-chengdu.myqcloud.com/IMG_3.png" alt="IMG_3" style="zoom: 50%;" />

- **QA && 常识推理：**

<img src="https://zlkqzimg-1310374208.cos.ap-chengdu.myqcloud.com/IMG_4.png" alt="IMG_4" style="zoom: 50%;" />

- **语义相似 && 分类任务：**

<img src="https://zlkqzimg-1310374208.cos.ap-chengdu.myqcloud.com/IMG_5.png" alt="IMG_5" style="zoom:50%;" />



### 1.6 消融实验

##### 1.6.1 迁移的decoder个数的影响
- 将预训练之后的模型的一部分decoder用于下游任务，得到结果：

<img src="https://zlkqzimg-1310374208.cos.ap-chengdu.myqcloud.com/IMG_6.png" alt="IMG_6" style="zoom: 60%;" />

由上图可知模型的精度和泛化能力会随着解码器层数增加而不断提升，而且目前还有提升空间

- **结论：**预训练得到的每个decoder都是对下游任务有作用的（个人觉得就是模型表达能力更加强大，并且不同的decoder所包含的知识是不同的）

##### 1.6.2 预训练的作用
- 作者去除了微调，以验证模型的zero-shot能力（没有进行过下游任务训练，而在下游的表现），并且和LSTM进行了比较（同样没有进行下游任务）：  

<img src="https://zlkqzimg-1310374208.cos.ap-chengdu.myqcloud.com/IMG_7.png" alt="IMG_7" style="zoom:60%;" />

- **结论：**生成式预训练任务是提升其语言建模能力，可以支持各种各样的下游相关任务。并且与 LSTM 相比，Transformer 的结构化注意力记忆有助于迁移

##### 1.6.3 其他实验
- 作者还探究了**微调时将LM作为额外目标的作用、将模型换为LSTM的对比、pre-training的作用**：

<img src="https://zlkqzimg-1310374208.cos.ap-chengdu.myqcloud.com/IMG_8.png" alt="IMG_8" style="zoom:50%;" />

- **结论：**
1. LM额外目标在大数据集上有提升，但是小数据集上没有
2. pre-training不可缺少





# 2 GPT v2

### 2.1 主要思想
- GPT2主要着眼于**只使用无监督的LM训练任务，来使模型具有zero-shot能力，不使用有监督数据微调，直接应用于下游任务**
- 本篇文章的核心观点就是：**只要无监督数据量足够大且足够多样，那么有监督任务就是无监督任务的子集。从一个尽可能大且多样化的数据集中一定能收集到不同领域不同任务相关的自然语言描述示例**
> Our approach motivates building as large and diverse a dataset as possible in order to collect natural language demonstrations of tasks in as varied of domains and contexts as possible.
- 举个例子：
> 1. 比如我在训练语言模型时，有一句话“The translation of word Machine Learning in chinese is 机器学习”，那在训练完这句话时，语言模型就自然地将翻译任务和任务的输入输出都学到了
> 2. 再比如，又碰到一句话“美国的总统是特朗普”，这一句话训练完，也就是一个小的问答了
> 3. 文章也给了用于训练的WebText Dataset中的英法互译真实实例：
>
> <img src="https://zlkqzimg-1310374208.cos.ap-chengdu.myqcloud.com/IMG_9.png" alt="IMG_9" style="zoom:50%;" />
- 还有一个需要注意的点是，在下游任务时，**由于预训练的预料都会自然、通顺的语言形式，所以下游任务的输入文本也需要重新构造为自然、通顺的形式**，如：
> 机器翻译任务：translate to french, { english text }, { french text }
> 阅读理解任务：answer the question, { document }, { question }, { answer }



### 2.2 训练目标

- GPT2的训练目标仍是LM，但是下游任务的建模发生了一些转变
- 一般的有监督任务是在估计分布：
$$
P(output|input)
$$
- 然而GPT2由于是要用同一个模型进行多任务，所以建模变为：
$$
P(output|input, task)
$$
对于output的估计还要基于具体是什么任务，相同的输入，不同的任务，所产生的output可能是不同的
- 针对不同任务，具体做法的话，就是上文提到的，将有监督数据构造为自然语言形式



### 2.3 模型结构

- 大体结构还是和GPT1一样，但是做了如下改动：
1. Layer Norm由每个sub-block之后，移到了之前：

<img src="https://zlkqzimg-1310374208.cos.ap-chengdu.myqcloud.com/IMG_11.png" alt="IMG_11" style="zoom:50%;" />

2. 在模型最后一个自注意力层之后，额外增加一个Layer Norm
3. 根据残差块的数量，减少了residual path所对应的权重，具体来说，模型一共有N个残差块，那么residual path的权重就都要乘$$1 / \sqrt{N}$$
4. 词汇量增加到50257，上下文大小从512增加到1024，batch size增加到512
- 模型结构大致如下：

<img src="https://zlkqzimg-1310374208.cos.ap-chengdu.myqcloud.com/IMG_10.png" alt="IMG_10" style="zoom:40%;" />



### 2.4 实验结果

- 在实验效果上，由于 GPT-2 主要是做 zero-shot，所以在实验部分，很多的实验对比都是在无监督的设定下进行的，也就是说他对比的都是无监督的算法

![image-20230329234220217](https://zlkqzimg-1310374208.cos.ap-chengdu.myqcloud.com/image-20230329234220217.png)

- GPT-2 在较多任务上对比无监督算法取得了一定的提升，证明了 zero-shot 的能力。但是，在很多任务上与有监督微调的方法相比还是有一些差距的



# 3 GPT v3

- GPT3不再像GPT2一样完全主推zero-shot，**而是采用few-shot，采用少量的有监督样本（一般10～100）来辅助模型进行推理**。但是，**GPT3采用有监督样本仅用于推理预测的时候，而不会进行微调的参数更新**



### 3.1 模型结构

- GPT3采用和GPT2一样的结构，**但是将其中的注意力机制变为了Sparse Attention**
- 传统的Attention是每个token之间两两计算attentino，复杂度为$$O(n^2)$$
- 而Sparse Attention除了相对距离不超过 k 以及相对距离为 k，2k，3k，... 的 token，其他所有 token 的注意力都设为 0，如下图所示：

<img src="https://zlkqzimg-1310374208.cos.ap-chengdu.myqcloud.com/IMG_14.png" alt="IMG_14" style="zoom: 80%;" />

其计算复杂度为$$O(n * \log n)$$
- Sparse Attention的好处：
> 1. **减少注意力层的计算复杂度**，节约显存和耗时，从而能够处理更长的输入序列
> 2. **具有“局部紧密相关和远程稀疏相关”的特性**，对于距离较近的上下文关注更多，对于距离较远的上下文关注较少
- 最后实验了不同规模的模型：

<img src="https://zlkqzimg-1310374208.cos.ap-chengdu.myqcloud.com/IMG_15.png" alt="IMG_15" style="zoom: 40%;" />



### 3.2 下游评估方法
- 具体到下游任务是，采用了三种不同的方法**（注意这三种方法都只用于推理预测，不会进行参数更新）**：
> 1. **Zero-shot：**仅使用当前任务的自然语言描述，不进行任何梯度更新
> 2. **One-shot：**当前任务的自然语言描述，加上一个简单的输入输出样例，不进行任何梯度更新
> 3. **Few-shot：**当前任务的自然语言描述，加上几个简单的输入输出样例，不进行任何梯度更新，也被称为**in-context learning（上下文学习）**
- **和fine-tune的对比：** 

<img src="https://zlkqzimg-1310374208.cos.ap-chengdu.myqcloud.com/IMG_16.png" alt="IMG_16" style="zoom: 40%;" />

Few-shot虽然和fine-tune一样都用到多个有监督数据，但是其数据量的需要较少（一般10～100个数据），摒弃不进行参数更新

- **三种方法对比的实验效果：**

<img src="https://zlkqzimg-1310374208.cos.ap-chengdu.myqcloud.com/IMG_17.png" alt="IMG_17" style="zoom: 40%;" />



### 3.3 训练数据

- GPT-3 使用了多个数据集，其中最大的是 Common Crawl，原始未处理的数据达到了 45TB，其实在 GPT-2 的时候他们就有考虑使用这个数据集，但是后来还是觉得这个**数据集太脏了**所以没用，但是现在 GPT-3 的模型规模太大了，使得训练对数据量的需求也增加了很多，他们不得不重新考虑这个数据集。因此，他们必须在这个数据集上做一些额外的数据清洗工作来尽量保证数据的质量
- **数据处理包括：**
1. 采用GPT2中的WebText、Wikiedia等高质量文本作为正样本，用Common Crawl中的样本作为负样本，训练一个LR二分类器，然后采用这个分类器对Common Crawl采样，只保留其中的正样本
2. 采用MinHashLSH算法，进行相似文本的去重，减少了大约10%的样本
3. 加入其他的高质量数据集，不同数据集是通过不同的权重进行采样：

<img src="https://zlkqzimg-1310374208.cos.ap-chengdu.myqcloud.com/IMG_18.png" alt="IMG_18" style="zoom:50%;" />



### 3.4 GPT3的局限性
1. 当生成文本长度较长时，GPT-3 还是会出现各种问题，比如重复生成一段话，前后矛盾，逻辑衔接不好等等；
2. 模型和结构的局限性，对于某一些任务，比如填空类型的文本任务，使用单向的自回归语言模型确实存在一定的局限性，这时候如果同时考虑上文和下文的话，效果很可能会更好一些；
3. 预训练语言模型的通病，在训练时，语料中所有的词都被同等看待，对于一些虚词或无意义的词同样需要花费很多计算量去学习，无法区分学习重点；
4. 样本有效性或者利用率过低，训一个模型几乎要把整个互联网上的文本数据全都用起来，这与我们人类学习时所需要的成本存在非常大的差异，这方面也是未来人工智能研究的重点；
5. 有一个不太确定的点是，模型到底是在“学习”还是在“记忆”？我们当然希望它能够学习，但是在使用数据量如此大的情况下，很难去判断它到底是什么样的；
6. 众所周知，GPT-3 的训练和使用成本都太大了；
7. GPT-3 跟很多深度学习模型一样，都是不可解释的，没办法知道模型内部到底是如何作出一系列决策的；
8. 训练数据中可能存在种族、性别等偏见，导致模型也会有这种偏见





# 4 InstructGPT

### 4.1 GPT存在的问题
- GPT的训练方式是采用LM的方法，是估计下一个时间步的词的概率分布：

<img src="https://zlkqzimg-1310374208.cos.ap-chengdu.myqcloud.com/IMG_19.png" alt="IMG_19" style="zoom:67%;" />

- 但是由于这是一个概率分布，所以模型的一些输入可能并不符合人类的预期，比如：

<img src="https://zlkqzimg-1310374208.cos.ap-chengdu.myqcloud.com/IMG_20.png" alt="IMG_20" style="zoom:67%;" />

- **对于上述问题的解决方案有两种：**
> 1. 在训练数据上，构造更加好的问答数据集，但是所花费的人工成本极大，因为训练数据集很大
> 2. 引入一个“老师”，让老师对GPT生成的回答进行打分排序，告诉模型人类更期望哪种结果**（这里的老师既可以是真人，也就是使用在线学习；也可以训练一个Reward Model来对模型结果自动打分排序）**



### 4.2 实现方案

- 模型通过**三个不同的数据集**，完成了三个子任务：

<img src="https://zlkqzimg-1310374208.cos.ap-chengdu.myqcloud.com/IMG_21.png" alt="IMG_21" style="zoom:55%;" />

1. **有监督微调（SFT）：**通过常见的prompt和labeler编写的response，来对GPT-3进行LM任务微调
2. **训练奖励模型（RM）：**通过常见的prompt，让SFT微调后的GPT模型生成多个response，labeler对这些response进行排序。再使用这些prompt+ response对，输入GPT进行打分
3. **强化学习（RL）：**只需要prompt，不需要有监督，采用PPO算法，再次微调SFT微调后的模型



### 4.3 训练数据

- 训练数据所用到的prompt来自两部分：
> 1. labeler先构造了一批prompt和对应的response，对GPT-3进行微调，然后上线内测
> 2. 将内测用户的prompt又收集起来，由labeler撰写response
- 然后将两部分数据分为三个子任务的数据集： 

<img src="https://zlkqzimg-1310374208.cos.ap-chengdu.myqcloud.com/IMG_22.png" alt="IMG_22" style="zoom: 67%;" />

注意，最后RL的数据集是只有prompt，且只包含收集的用户的prompt，而SFT和RM是两者都有



### 4.4 实现细节

##### 4.4.1 SFT
- 方法很简单，就不多赘述
- 值得注意的是，作者在SFT中一共训练了16个epoch，但是发现在第一个epoch后就过拟合了（这么大的模型用这么小的数据肯定过拟合）。**但是由于这个模型并不是微调完就直接拿来用，所以过拟合也没关系。甚至更多的epoch甚至能产生更高的RM分数的输出**



##### 4.4.2 RM

- 先采用SFT后的模型，对一个prompt生成多个response，并对每一对prompt+response，让labeler进行排序：

<img src="https://zlkqzimg-1310374208.cos.ap-chengdu.myqcloud.com/IMG_23.png" alt="IMG_23" style="zoom: 25%;" />

- 然后将SFT后的模型最后的输出层去掉，转而变为一个只有一个神经元的线性层
- 将每一对prompt+response连结起来，输入该模型，最后输出相当于两者契合的logit分数。然后采用以下损失函数进行优化：
$$
\operatorname{loss}(\theta)=-\frac{1}{\left(\begin{array}{c}
K \\
2
\end{array}\right)} E_{\left(x, y_{w}, y_{l}\right) \sim D}\left[\log \left(\sigma\left(r_{\theta}\left(x, y_{w}\right)-r_{\theta}\left(x, y_{l}\right)\right)\right)\right]
$$
其中，K是每个prompt生成的response数量，$$y_w, y_l$$分别是prompt输入x的输出response，且$$rank(y_w) \ge rank(y_l)$$，外层函数就相当于一个Logestic Regression
- RM是采用的6B的模型，因为作者发现**大模型（比如175B）训练后期loss不稳定**
- 此外，作者还提出了另一种方法：采用交叉熵，将排名第一的输出当作正样本，其他输出当作负样本，但是**非常容易过拟合**



##### 4.4.3 RL

- RL涉及三个模型：RM模型$$r_{\theta}$$、SFT模型$$\pi^{SFT}$$和我们最终想要得到的RL模型$$\pi^{RL}$$；以及两个数据集RL自身的数据集$$D_{RL}$$和预训练时的一部分数据集$$D_{pretrain}$$
- 优化目标如下：
$$
\begin{aligned}
\operatorname{objective}(\phi)= & E_{(x, y) \sim D_{\pi_{\phi}^{\mathrm{RL}}}}\left[r_{\theta}(x, y)-\beta \log \left(\pi_{\phi}^{\mathrm{RL}}(y \mid x) / \pi^{\mathrm{SFT}}(y \mid x)\right)\right]+ \\
& \gamma E_{x \sim D_{\text {prerrain }}}\left[\log \left(\pi_{\phi}^{\mathrm{RL}}(x)\right)\right]
\end{aligned}
$$
- 一开始$$\pi^{RL}$$是由$$\pi^{SFT}$$初始化得来的
- 对于第一项，是想让$$\pi^{RL}$$的输出得到的RM分数尽可能高，并且在这个微调过程中，$$\pi^{RL}$$和$$\pi^{SFT}$$的差距不能过大，所以减去两者的KL散度来保证这个差距
- 如果只使用第一项，方法就称作PPO。但是为了防止模型遗忘预训练时的知识，引入第二项，也就是预训练任务的优化目标，加入第二项后则称为PPO-ptx



### 4.5 实验结果

<img src="https://zlkqzimg-1310374208.cos.ap-chengdu.myqcloud.com/IMG_24.png" alt="IMG_24" style="zoom: 80%;" />




# 5 GPT vs BERT
- 编码器和解码器的选取倒不是 GPT 和 BERT 的区别，它们的区别主要是预训练目标函数的选取，有人认为 GPT 选择的是一个更难的训练目标，它是根据前面的信息去预测下文，预测未来肯定是比完形填空难度要更大的。这也能从某种程度上解释了为什么相同规模的 GPT 和 BERT 模型，GPT 的效果要比 BERT 差。
- 但是从另一个角度去想，如果能够把预测未来这个事情做好的话，它最终所能达到的效果的天花板一定是更高的，这可能也是 OpenAI 从一开始到现在一直坚持使用标准语言模型目标函数来做预训练模型的其中一个原因吧，当然这只是一种猜想。事实证明，从 GPT-3 开始，到最近的 ChatGPT，OpenAI 所取得的令人惊艳的效果也一定程度上证明了他们的选择的正确性。


