---
title: RoBERTa总结
math: true
date: 2022-8-9
---



- 作者评估了BERT中的各种超参和数据集大小以及训练策略等方面，发现**BERT是训练不足的**



# 1  RoBERTa的主要改变

- 作者提出了RoBERTa（A Robustly Optimized BERT Pretraining Approach），是BERT的变体，主要的更改如下：

> 1. 用更多的数据和更大的batch size，训练更长的时间
> 2. 去除了NSP
> 3. 在更长的句子上训练
> 4. 进行动态的mask

- 在优化器方面也有细微改变，改变了最大学习率和warmup steps，并且把$$\beta_2$$改为了0.98，在更大的batch size上训练时，得到了更稳定的结果





# 2 训练策略的改变

### 2.1 动态和静态mask

- **两种mask策略：**

> 1. 静态mask：原BERT中使用的方法，在数据预处理时，进行mask。在本实验中，为了避免出现每个epoch的mask的位置都相同，在40个epoch中进行了10次随机mask，这样每个mask实例只会出现4次
> 2. 动态mask：在每次数据喂入模型时进行mask，这样每次mask的位置都不同

- 实验结果如下：

<img src="https://zlkqzimg-1310374208.cos.ap-chengdu.myqcloud.com/image-20220809143220226.png" alt="image-20220809143220226" style="zoom:80%;" />

- **动态mask相对于静态mask得到了细微的提升（稍微有丁点卵用）**



### 2.2 模型的输入形式和NSP任务

- **使用了4种输入策略：**

> 1. **SEGMENT-PAIR+NSP：**原BERT使用的方法，**输入是一个segment对，每个segment可以包括多个句子**，只要总长度小于512就行，带有NSP Loss
> 2. **SENTENCE-PAIR+NSP：**和前者类似，**只是每个segment只能是单个句子**，同样带有NSP Loss。由于两个单句子一般长度都远远不足512，为了实验公平，作者适当的增加了其实验数据
> 3. **FULL-SENTENCES：输入是从一个或多个文档中采样的一个full-sentence（可由多个句子组成）**，当采样到文档的最后时，可以继续采下一个文档的句子，但是需要在中间加一个额外的分隔token。**无NSP Loss**
> 4. **DOC-SENTENCES：和前者类似，但是不能跨文档采样，无NSP Loss**。在靠近文档末尾采样，句子长度会较短，所以同样为了实验公平，适当的增加了数据

- 实验结果如下：

<img src="https://zlkqzimg-1310374208.cos.ap-chengdu.myqcloud.com/image-20220809150834375.png" alt="image-20220809150834375" style="zoom:75%;" />

- 将前两种策略进行对比，**可以发现使用单个句子会降低表现，推测原因为：这样模型无法学到长范围的依赖关系**

> We find that using individual sentences hurts performance on downstream tasks, which we hypothesize is because the model is not able to learn long-range dependencies.  

- 再对比前两种和后两种策略，**可以发现移除NSP任务，也可以得到细微的提升**

> Removing the NSP loss matches or slightly improves downstream task performance.

- 对比后两种策略，**可以发现不跨文档比跨文档稍好一点**



### 2.3 使用更大的Batch Size

- 作者使用相同的数据量，但是不同的batch size，来探究batch size的影响，结构如下：

<img src="https://zlkqzimg-1310374208.cos.ap-chengdu.myqcloud.com/image-20220809152515016.png" alt="image-20220809152515016" style="zoom:80%;" />

图中bsz为batch size，ppl为困惑度



### 2.4 分词方式

- 原版BERT使用char-level的BPE（也就是wordpiece），而本文使用byte-level的BPE，增加了词典大小，和增加了一些参数量，表现还有细微的下降😅，但是基本相同

> - 基于 char-level ：原始 BERT 的方式，它通过对输入文本进行启发式的词干化之后处理得到。
>
> - 基于 bytes-level：与 char-level 的区别在于bytes-level 使用 bytes 而不是 unicode 字符作为 sub-word 的基本单位，因此可以编码任何输入文本而不会引入 UNKOWN 标记。
>
>   
>
> - 当采用 bytes-level 的 BPE 之后，词表大小从3万（原始 BERT 的 char-level ）增加到5万。这分别为 BERT-base和 BERT-large增加了1500万和2000万额外的参数





# 3 对比试验

- 对于实验用于探究预训练数据量的多少和训练时间对表现的影响，实验采用Large的模型规模，并且RoBERTa采用：动态mask + FULL_SENTENCES without NSP + 大batch size + byte-level BPE，实验结果如下：

<img src="https://zlkqzimg-1310374208.cos.ap-chengdu.myqcloud.com/image-20220809164004895.png" alt="image-20220809164004895" style="zoom:75%;" />

- **可以看到在相同数据量条件下，RoBERTa相比于BERT有较大提升**

- **并且增加数据量可以得到显著的提升，增加训练时间同样可以得到细微提升，并且没有造成过拟合**





# 4 在特定下游任务的表现

### 4.1 GLUE

- 实验考虑了$$batch\_size \in \{16, 32\}$$，$$lr \in \{1e-5, 2e-5, 3e-5\}$$，并在前6%的steps使用线性warm up，之后使用线性衰减至0，使用10个epochs，但是设置了early stop，实验结果如下：

<img src="https://zlkqzimg-1310374208.cos.ap-chengdu.myqcloud.com/image-20220809170036407.png" alt="image-20220809170036407" style="zoom:75%;" />



### 4.2 SQuAD

- 在本实验中，XLNET和BERT都加上了QA数据集，但是RoBERTa仅使用了SQuAD。并且XLNET使用了逐层不同的学习率，结果如下：

<img src="https://zlkqzimg-1310374208.cos.ap-chengdu.myqcloud.com/image-20220809170501845.png" alt="image-20220809170501845" style="zoom:70%;" />



### 4.3 RACE

- 一个做阅读理解的数据集，让模型从4个答案中选出一个最合适的。结果如下：

<img src="https://zlkqzimg-1310374208.cos.ap-chengdu.myqcloud.com/image-20220809170610509.png" alt="image-20220809170610509" style="zoom:80%;" />





# 5 超参

- **预训练：**

<img src="https://zlkqzimg-1310374208.cos.ap-chengdu.myqcloud.com/image-20220809170735173.png" alt="image-20220809170735173" style="zoom:50%;" />

- **微调：**

<img src="https://zlkqzimg-1310374208.cos.ap-chengdu.myqcloud.com/image-20220809170753600.png" alt="image-20220809170753600" style="zoom:67%;" />