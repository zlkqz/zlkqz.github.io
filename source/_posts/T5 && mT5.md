---
title: T5 && mT5
math: true
date: 2022-11-22
---



- T5（Text-to-Text Transfer Transformer）模型采用了一种Text-to-text（文本到文本）的框架，想要把NLP领域的许多常见任务，如文本分类、QA等，都套到这个框架中解决
- 如机器翻译任务，输入”translate English to German: That is good.”，目标输出是”Das ist gut.”，在输入中” : “前面称为prompt，代指现在需要执行的任务
- 这样的好处是可以把所有的问题都套进去一个统一的范式，从而可以采用同样的模型架构、同样的训练策略、同样的损失函数、同样的解码手段。

<img src="https://zlkqzimg-1310374208.cos.ap-chengdu.myqcloud.com/image-20221126211139471.png" alt="image-20221126211139471" style="zoom:80%;" />

- 这里我们讲的是T51.0，之后再介绍T51.1的不同点



### 1.1 C4数据集

- C4全称Colossal Clean Crawled Corpus，跟GPT2、GPT3的训练数据来源一样，是从网上爬的文本数据，由于是爬的数据，所以数据量足够大，而且类型丰富，缺点是数据质量差，需要过滤，过滤手段包括
- 经过一番数据清洗后得到了一个750G的数据集



### 1.2 Baseline

- 其实T5就是经过很多个实验，对不同的模型结构、训练策略等进行对比，然后挑出效果最好的，所以我们先给出实验中用于对比的baseline
- **模型结构：**和Transformer一模一样，12层的encoder-decoder架构
- **训练/预测策略：**训练时采用teacher-forcing，预测时采用贪婪搜索
- **预训练：**在C4上面训练$$2^{19}$$个steps，batch_size=128，seq_len=512。**预训练并没有覆盖所有C4数据集，即没一个样本会重复训练**。预训练目标稍后介绍
- **学习率调整：**采用平方根倒数：

$$
l r=\frac{1}{\sqrt{\max (n, k)}}, k=10^{4}
$$

- **微调：**对每个下游任务训练$$2^{18}$$个steps
- **词表：**采用WordPiece，大约有32000个token，有部分非英语词



### 1.3 无监督预训练目标

- 预训练目标和BERT一样，都是采用随机mask破坏文本，然后通过上下文将这个词训练出来，称为**Denoising**的预训练目标
- **对输入随机挑选15%的token，然后使用一个哨兵token进行替换，注意挑选出来的token如果时连续的text span，则只用一个哨兵token进行替换。然后target文本变为：每个哨兵token+其对应的值的形式，最后再接一个特殊的哨兵token，表示结束**
- 举例栗子：

![image-20221126223744785](https://zlkqzimg-1310374208.cos.ap-chengdu.myqcloud.com/image-20221126223744785.png)

如上图，将for inviting和last分别替换成了两个不同的哨兵token，然后target变为了分别预测每个哨兵token，然后文本最后预测出另一个哨兵token\，表示结束



### 1.4 不同模型结构的对比

- 针对self-atttion，有三种mask方式：

![image-20221126230434585](https://zlkqzimg-1310374208.cos.ap-chengdu.myqcloud.com/image-20221126230434585.png)

分别为：

> 1. fully-visible：每个时间步都对其他时间步可见
> 2. causal：对未来的时间步不可见
> 3. causal with prefix：前面两者的结合，prefix部分的token能看到prefix所有token的信息，非prefix的token只能看到它的上文信息。那么什么是prefix，如上面提到的英文翻译德文的例子，prefix就是”translate English to German: That is good.”，说白了就是输入部分的时间步是fully-visible，输出部分的时间步是causal

- 针对三种不一样的mask方式，作者对如下三种模型架构进行了比较：

![image-20221126230815635](https://zlkqzimg-1310374208.cos.ap-chengdu.myqcloud.com/image-20221126230815635.png)

- 对此，作者提出了五种模型（为了公平，只有两个模型有同样数量的参数或者同样的计算效率才能进行比较）：

> 1. **Encoder-decoder：**编码层和解码层各有L层
> 2. **Enc-dec, shared：**编码层和解码层各有L层，但它们参数共享，所以参数减半
> 3. **Enc-dec, 6 layers：**编码层和解码层各有L/2层
> 4. **LM：**只有L层解码层，采用语言模型的形式
> 5. **Prefix LM：**只有L层解码层，但采用Prefix语言模型的形式

- 并且还对比使用了两种预训练目标：

> 1. **Denoising：**即baseline中使用的随机mask词然后预测出来
> 2. **LM：**LM中常用的自回归预测，即每个时间步预测通过上个时间步的输出来进行当前时间步的输出预测

- 最后得到以下结果：

<img src="https://zlkqzimg-1310374208.cos.ap-chengdu.myqcloud.com/image-20221127105905994.png" alt="image-20221127105905994" style="zoom:67%;" />



### 1.5 不同的无监督预训练目标对比

- 首先介绍采用的预训练目标：

<img src="https://zlkqzimg-1310374208.cos.ap-chengdu.myqcloud.com/image-20221127110614966.png" alt="image-20221127110614966" style="zoom: 80%;" />

> 1. **Prefix LM：**输入前部分文本，预测剩余的文本
> 2. **BERT-style：**利用和BERT中一样的mask策略，然后预测出原文本
> 3. **Deshuffling：**随机打乱文本，然后预测出原文本
> 4. **MASS-style：**和BERT-style的不同在于，mask时直接用[M]替换
> 5. **noise, replace spans：**前文提到的无监督预训练目标
> 6. **noise, drop tokens：**和5差不多，但是不用哨兵token替换，直接drop
> 7. **Random spans：**和5差不多，但是每次选择的是一个长为3的text span

- 作者首先对前三种目标进行对比：

<img src="https://zlkqzimg-1310374208.cos.ap-chengdu.myqcloud.com/image-20221127112236904.png" alt="image-20221127112236904" style="zoom:80%;" />

- 结果发现BERT-style效果最好，然后再使用余下的方法和BERT-style进行比较：

<img src="https://zlkqzimg-1310374208.cos.ap-chengdu.myqcloud.com/image-20221127112315271.png" alt="image-20221127112315271" style="zoom:80%;" />

**其实这三种和BERT-style差不多，但是后两种不需要预测出整个原文本，更快，**

- 此外，作者还对比了**不同的文本corruption率和允许的最长text span的长度**（由于连续的mask掉的token都处理为一个哨兵token，允许最长的text span即指最多只有3个token可以替换成一个哨兵token，超过三个要使用另一个哨兵token）

<img src="https://zlkqzimg-1310374208.cos.ap-chengdu.myqcloud.com/image-20221127112647149.png" alt="image-20221127112647149" style="zoom:80%;" />

<img src="https://zlkqzimg-1310374208.cos.ap-chengdu.myqcloud.com/image-20221127112657299.png" alt="image-20221127112657299" style="zoom:80%;" />

- 最后对这部分实验做个总结，作者是逐层递进来进行的对比试验：

<img src="https://zlkqzimg-1310374208.cos.ap-chengdu.myqcloud.com/image-20221127112910646.png" alt="image-20221127112910646" style="zoom:80%;" />



### 1.6 数据集对比

- 作者还对不同类型和不同大小的预训练数据集进行了对比
- 最后得出：**用更专的数据来做预训练，对下游任务的提升越明显，或者换句更准确的话来说，预训练的语料跟任务语料domain越接近，效果越好，并且数据越多越好，即使预训练不能覆盖完**
- 所以个人认为最佳的策略是**在丰富的数据上进行预训练，然后再在领域相关、任务相关的语料上继续预训练，最后再fine-tuning**



### 1.7 训练策略

- **fine-tuning方法：**作者对三种微调方法进行了对比：

> 1. **All parameters：**微调时更新所有参数
> 2. **Adapter layers：**adapter layers接在编码器和解码器的每一个block的全连接层后面，在fine-tuning的时候只更新它们。adapter layers有一个内部维度d作为超参
> 3. **Gradual unfreezing：**一开始离任务层近的参数先更新，其它保持不动，随着训练的进行，逐渐放开其它层的参数。

实验结果：

<img src="https://zlkqzimg-1310374208.cos.ap-chengdu.myqcloud.com/image-20221127115209871.png" alt="image-20221127115209871" style="zoom:80%;" />

发现还是更新所有参数效果最好，但是会慢很多

- **多任务学习：**得益于提出的text-to-text范式，我们可以**在预训练的时候把有监督的训练也加进来，一起做预训练（注意：多任务学习预训练中的数据集包括原本的无监督数据集+多个有监督数据集）**。现在问题就变为了**给定多个不同任务的数据集，怎样对数据进行采样**，作者使用了以下三种策略：

> 1. **Examples-proportional mixing：**设第$$i$$个任务的数据集大小为$$e_i$$，那么采样自第$$j$$个数据集的概率为$$r_j=\min(e_j,K)/∑_i\min(e_i,K)$$，其中K为提前设置好的超参
> 2. **Temperature-scaled mixing：**在上面的策略下，再做一些软化，具体来说就是求得$$r_j$$后再开1/T方根，T为提前设置好的超参，T越大，各个任务数据集采样越均衡
> 3. **Equal mixing：**各数据集均匀采样

实验结果：

<img src="https://zlkqzimg-1310374208.cos.ap-chengdu.myqcloud.com/image-20221127120459146.png" alt="image-20221127120459146" style="zoom:80%;" />

实验结果都一般，**不过注意，这里的多任务学习是多个任务一起做训练，相当于把pre-training和fine-tuning两个合并了，而不会对单个任务进行fine-tuning**，所以效果不好也可以理解。

- **多任务学习+fine-tuning：**作者采用了一下集中训练策略进行比较：

> 1. **Unsupervised pre-training + fine-tuning：**baseline中使用的方法，先无监督预训练再在特定的下游任务上微调
> 2. **Multi-task training：**直接在多任务数据集上训练（注意mutl-task的训练集中有有监督的也有无监督的）
> 3. **Multi-task pre-training + fine-tuning：**多任务预训练+微调
> 4. **Leave-one-out multi-task training：**在预训练的时候同样使用多任务，但是要去除和下游任务相关的那个数据集，然后再在下游任务微调
> 5. **Supervised multi-task pre-training：**在多任务预训练的时候把无监督任务剔除掉，然后再微调

实验结果：

![image-20221128120902619](https://zlkqzimg-1310374208.cos.ap-chengdu.myqcloud.com/image-20221128120902619.png)

通过实验结果得到以下结论：

> 1. 使用Multi-task pre-training + fine-tuning的结果和baseline差不多，表明**在多任务后使用微调可以减轻不同的数据集mixing比例之间的权衡**，即你选用的Mixing方式（比如上文的三种）不一定是最好的，但是微调可以减轻这种错选带来的干扰
> 2. Leave-one-out的效果只有一点点下降，表明**多任务学习不会导致严重的任务干扰**
> 3. 使用Supervised multi-task pre-training几乎除了翻译任务都导致了下降，表明**翻译任务从英语预训练中学得的很少，反之其他任务仍然很依赖无监督预训练**



### 1.8 Scaling

- 此外，作者还对模型规模等进行了测试，得出**使用更多的数据、训练更大的模型、模型融合都能提高性能**
- 最后提一句，经过一系列实验，T5还是选择了**Multi-task pre-training + fine-tuning**以及预测时采用束搜索，无监督预训练目标采用了**noise, replace spans**





# 2 T5 v1.1

- 上文讲的是T5 v1.0，谷歌之后又发布了一个T5 v1.1，只有一些细微差别，改进如下：

> 1. 前馈神经层的激活函数由ReLU改为了GEGLU
> 2. 在pre-training的时候关闭Dropout，在微调的时候重新开启
> 3. 预训练的时候只使用C4数据集，而不混入下游数据集
> 4. Embedding层和最后的分类层没有使用Weight Tying
> 5. 模型形状有点不同，较大的 d_model 和较小的 num_heads 和 d_ff





# 3 mT5

- mT5的预训练目标和策略等等和T5基本相同， 值得注意的是mT5使用的是T5 v1.1



### 3.1 mC4数据集

- 一个多语言版的C4数据集，但是使用的数据清洗方法和T5不同：
- 对于多语言模型，一个很重要的部分是如何多多种语言进行采样，**不同语种数据占比不同，有的语言样本少（low-resource languages ），如果不常采样到，模型就会由于样本过少而过拟合；如果样本量太大（high-resource languages ），内容丰富，模型又可能欠拟合，所以不能让模型遍历太多high-resource languages**
- 要解决上述问题，直观上来说可以使用均匀分布来采样，但是使用均匀分布效果肯定比较差，因为很多high-resource languages 压根用不到
- 所以采用了：

$$
P(L) \propto L^{\alpha}
$$

其中L为对应语言的样本数，$$\alpha \in [0,1]$$为超参，$$\alpha$$越小分布越接近均匀分布，**mT5经过实验发现$$\alpha=0.3$$最合适**。那么这样就可以**适当提升low-resource languages的采样概率而适当减少high-resource languages的采样概率**

- mC4中不同语言的样本数，以及使用不同$$\alpha$$的采样概率：

<img src="https://zlkqzimg-1310374208.cos.ap-chengdu.myqcloud.com/image-20221129191709077.png" alt="image-20221129191709077" style="zoom:80%;" />



### 3.2 微调策略

- 模型在mC4上预训练之后，作者采用了一下三种微调方式进行对比（微调采用lr = 0.001）：

> 1. **zero-shot：**仅在英语训练集上微调
> 2. **translate-train：**在英语+由英语翻译到所有目标语言的数据集上微调
> 3. **in-language multitask：**在目标语言的gold data上微调（这里是真实的人工表述的数据，而tanslate-train的目标语言数据是翻译过来的）

结果如下：

<img src="https://zlkqzimg-1310374208.cos.ap-chengdu.myqcloud.com/image-20221129193118134.png" alt="image-20221129193118134" style="zoom:80%;" />

- 此外，作者还对比了采用不同的模型参数量对这三种微调方式的提升：

![image-20221129193253828](https://zlkqzimg-1310374208.cos.ap-chengdu.myqcloud.com/image-20221129193253828.png)



### 3.3 T5 vs mT5

- 作者还对比了T5和mT5在英语QA任务上的效果差异：

<img src="https://zlkqzimg-1310374208.cos.ap-chengdu.myqcloud.com/image-20221129193755886.png" alt="image-20221129193755886" style="zoom:80%;" />

- 发现**mT5还是略逊色于T5，但是随着模型规模的增大，之间的差异越来越小。也证明了多语言模型同样有能力比肩单语言模型**



### 3.4 消融实验

- 作者还对训练的各方面进行了一些消融实验，策略如下：

> 1. **Dropout 0.1：**由于使用的是T5 v1.1，所以在预训练时没有使用Dropout，这里为了对照又把Dropout加上了
> 2. **Sequence length 512：**将最大序列长度减少为512
> 3. **Span length 10：**将连续token的长度由3变为10
> 4. **$$\alpha=0.7,0.2$$：**采样时的超参改一下
> 5. **No line length filter：**数据清洗时的策略改一下
> 6. **Add Wikipedia data：**预训练使用mC4+Wikipedia data



### 3.5 zero-shot微调策略的问题

- 采用zero-shot会造成预测时产生一些非法输出：

> 1. **Normalization：**prediction是合法的，但是unicode characters被替代了，可以通过Unicode NFKC normalization来恢复
> 2. **Grammatical adjustment：**answer本身就存在语法问题
> 3. **Accidental translation：**模型直接做了翻译，将目标语言翻译成英文了，以至于生成部分或者完整英文
>
> 同时，在一些短语生成的时候，出现正确答案之前可能会先预测出两个英语词
>
> 上面最常出现的是Accidental translation

以下是非法输出的一些栗子：

<img src="https://zlkqzimg-1310374208.cos.ap-chengdu.myqcloud.com/image-20221129195942926.png" alt="image-20221129195942926" style="zoom:80%;" />

- **产生原因：**模型在微调的时候压根没有接触过non-English的target文本，在non-English上做推理时，non-English的likelihood会降低，以至于English变成最可能的输出
- **解决方法：**在微调时再次使用**少量的mC4数据进行无监督二次预训练**（和微调的样本数比例是1：100，并且包含全部101种语言），并且二次预训练时**删除了target文本中的哨兵token**，因为最后的结果发现在下游任务时就偶尔会预测出哨兵token，然后还将α从0.3降为0.1，**使采样分布十分近似于均匀分布**。结果提升显著：

<img src="https://zlkqzimg-1310374208.cos.ap-chengdu.myqcloud.com/image-20221129201243017.png" alt="image-20221129201243017" style="zoom:80%;" />