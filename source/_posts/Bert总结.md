---
title: BERT总结
math: ture
date: 2022-2-25
---

- BERT是一种通过**在预训练时使用无监督**方法能在**每一层**实现**双向**表征的语言模型，并且使用微调的方法，在**具体的下游任务时不需要task-specific architecture**，只需要添加一两层和少部分参数，十分易于迁移

- BERT带来的主要提升是解决了双向性的问题。如OpenAI GPT使用的是left-to-right（LTR）Transformer结构，失去了双向性。又比如ELMo使用简单的将left-to-right（LTR）的LSTM和right-to-left（RTL）的LSTM在最后简单的连结来实现双向性，而BERT能在每一层都实现双向，并且相比于在最后简单的连结更具有直观性和可解释性。





# 1 BERT的结构

### 1.1 结构和规模

- BERT的结构十分简单，就是由**多个Transformer的encoder组合而成**
- 我们将encoder的数量设为L，隐藏层的单元数设为H，自注意力头的个数设为A，则BERT可分为$$BERT_{BASE}$$（L=12，H=768，A=12，总参数量=110M  ）和$$BERT_{LARGE}$$（L=24，H=1024，A=16，总参数量=340M）两个版本

- $$BERT_{LARGE}$$在几乎所有的任务上都是优于$$BERT_{BASE}$$的，特别是特别小的数据集上



### 1.2 BERT的输入输出

- BERT使用WordPiece embeddings  

- BERT的**输入可以是一个句子也可以是两个句子**，每个输入的**最开始都需要加一个[CLS] token**，如果输入包含两个句子（sentence A and sentence B），则**中间需要加入一个[SEP] token来做分隔**
- **总的输入为**：对应的token embedding+segment embedding+position embedding的总和：

![image-20220701114959234](https://zlkqzimg-1310374208.cos.ap-chengdu.myqcloud.com/image-20220701114959234.png)

其中segment embedding是用以区分sentence A和sentence B（**第一个句子的segment embedding都是0，第二个的都是1**），而position embedding和Transformer中的不一样，Transformer是采用三角函数，而**BERT采用learned position embedding**

- 输入输出的形式大致如下：

![image-20220701115457939](https://zlkqzimg-1310374208.cos.ap-chengdu.myqcloud.com/image-20220701115457939.png)

**其中C为[CLS]对应的最终的embedding，在分类任务时作为整个序列的总表征（但是C在微调之前是没有具体意义的向量，因为他是通过NSP预训练出来的）**。$$T_i$$为第$$i$$个token所对应的embedding





# 2 BERT的Pre-training

- BERT的Pre-training可分为MLM和NSP，分别对应token级的任务和sentence级的任务
- Pre-training采用的是**无监督的方法**
- 在Pre-training数据的选择上，使用document-level corpus要优于shuffled sentence-level corpus



### 2.1 Masked Language Model（MLM）

#### 2.1.1 MLM的输入

- 每个输入的sequence会**随机mask掉15%的token**，并且在最后预测mask掉的地方是什么词（通过将该token最后对应的embedding送入softmax层并采用交叉熵损失，分类个数为整个词典的token数）
- **其中mask的策略为**，对于一个要mask的token：

1. 80%的概率变为[MASK]
2. 10%的概率变为随机词
3. 10%的概率不变

举个栗子：

<img src="https://zlkqzimg-1310374208.cos.ap-chengdu.myqcloud.com/image-20220701122513870.png" alt="image-20220701122513870" style="zoom:80%;" />



#### 2.2 采用此mask策略的原因

- 预训练时是有[MASK]的，但是微调时是没有的，那么**微调时模型就只能根据其他token的信息和语序结构来预测当前词，而无法利用到这个词本身的信息**（因为它们从未出现在训练过程中，等于模型从未接触到它们的信息，等于整个语义空间损失了部分信息），所以会**产生预训练和微调的mismatch**
- 而保留下来的信息**如果全部使用原始token，那么模型在预训练的时候可能会偷懒，直接照抄当前的token**，所以需要随机换成其他词，会让模型不能去死记硬背当前的token，而去**尽力学习单词周边的语义表达和远距离的信息依赖**，尝试建模完整的语言信息
- 但是随机替换不能太多，要不然肯定会对模型产生误导，以下是经过多次实验的数据：

![image-20220701123511287](https://zlkqzimg-1310374208.cos.ap-chengdu.myqcloud.com/image-20220701123511287.png)

可以看到只使用随机替换，会对结果产生极大的影响



#### 2.3 MLM的问题

- 由于MLM每次只mask掉15%的词，所以只预测15%的词，所以需要更多的steps才能收敛，以下是MLM和LTR模型的对比：

![image-20220701123837335](https://zlkqzimg-1310374208.cos.ap-chengdu.myqcloud.com/image-20220701123837335.png)

可以看到MLM收敛速度更慢，需要更多的steps，但是所获得的改进远大于所增加的成本，所以问题不大



### 2.2 Next Sentence Predictoin（NSP）

- NSP的输入为两个句子，有50%的概率sentence B是sentence A的下一句，有50%的概率不是，举个栗子：

<img src="https://zlkqzimg-1310374208.cos.ap-chengdu.myqcloud.com/image-20220701124711635.png" alt="image-20220701124711635" style="zoom:90%;" />

- 在最后使用C向量送入下一层，判断为IsNext or NotNext



### 2.3 Pre-training的细节

- **优化：**
> 1. Adam（learning rate = $$10^{-4}$$，$$\beta_1 = 0.9$$， $$\beta_2 = 0.999$$）
> 2. learning rate在前10000个steps采用warmup，并且还应用了线性衰减
> 3. 0.01的L2权重衰减和0.1的Dropout
> 4. batch size = 256 sequences / batch

- 激活函数采用gelu而非relu
- 损失为MLM的最大似然和NSP的最大似然的和

- 由于attention是随序列长度进行平方增长的，所以为了提高预训练速度，在实验时，**先在90%的steps应用应用128的序列长，然后在剩下的10%的steps中改为512序列长度，来学习position embedding**






# 3 BERT的Fine-tuning

### 3.1 Fine-tuning的一般做法

- 都是在最后加上一两层，来进行微调。对于Transformer Encoder的输出：
1. 如果是token级的下游任务，如sequence tagging和question answering，是直接将对应的token输出的embedding送入下一层。
2. 如果是sentence级的下游任务，如sentiment analysis，需要将[CLS]对应的输出，也就是C，送入下一层用以分类



### 3.2 Fine-tuning的细节

- 大多数超参数和pre-training时是一样的，除了batch size、learning rate和epochs
- dropout的概率还是保持为0.1
- 在实验中发现，以下几个超参的选择，适用于大多数的任务：

> **Batch size：16， 32**
>
> **Learning rate (Adam)：$$5 \times 10^{-5}, 3 \times 10^{-5}, 2 \times 10^{-5}$$  **
>
> **Number of epochs：2，3，4  **

- 并且还发现大数据集相比小数据集对于超参的选择是不那么敏感的





# 4 BERT实践

- 下面介绍BERT在各种下游任务上的表现：

<img src="https://zlkqzimg-1310374208.cos.ap-chengdu.myqcloud.com/image-20220701233158369.png" alt="image-20220701233158369" style="zoom:70%;" />

### 4.1 GLUE

- GLUE全称为The General Language Understanding Evaluation，包含了各种而样的自然语言理解任务
- 在BERT中我们只添加了一个多分类输出层，将[CLS]对用的输出C，送入该层，再使用softmax，计算损失
- 采用的超参：batch size = 32，epochs = 3，learning rate =  $$5 \times 10^{-5}，4 \times 10^{-5}，3 \times 10^{-5}，2 \times 10^{-5}$$

- 在微调时，**$$BERT_{LARGE}$$在小数据集上的结果是不稳定的**，所以采取了**多次随机重启**（不一样的数据重洗和分类层的参数初始化），并且选择了在验证集上结果最好的模型



### 4.2 SQuAD 

- SQuAD全称The Stanford Question Answering Dataset，收录了100k的QA对，其中每个Query的Answer是在对应的Passage中的一段连续文本（answer span）

#### 4.2.1 SQuAD v1.1

- 首先设$$S \in R^H$$和$$E \in R^H$$分别为answer span中的第一个词和最后一个词的embedding
- 那么$$word_i$$作为第一个词的概率，可以使用点积+softmax的求得：（其中$$T_i$$是$$word_i$$对应的output）

$$
P_{i}=\frac{e^{S \cdot T_{i}}}{\sum_{j} e^{S \cdot T_{j}}}
$$

将$$word_i$$作为最后一个词的概率也是一样的算法，只是把S替换成E

- 在训练时的损失为正确的开始和结束位置的最大似然

- 在预测时，每个候选位置，即将$$word_i$$到$$word_j$$作为answer的score为：

$$
S \cdot T_i + E \cdot T_j \quad (j \geq i)
$$

然后取最大score的侯选位置作为输出

- 超参：batch size = 32，epochs = 3，learning rate = $$5 \times 10^{-5}$$

- 在具体实验中，应用于SQuAD数据集上前，先在TriviaQA上微调，进行适当的数据增强



#### 4.2.2 SQuAD v2.0

- SQuAD v2.0相对于SQuAD v1.1增加了一个No Answer的输出，因为一个问题的答案并不总是出现在passage中的，No Answer的的具体形式为start和end都是[CLS]的answer span，预测为No Answer的score为：

$$
s_{null} = S \cdot C + E \cdot C
$$

当满足下式时，则不预测为No Answer：
$$
\hat{s_{i, j}}>s_{\mathrm{null}}+\tau
$$
其中$$\hat{s_{i, j}}=\max _{j \geq i} S \cdot T_{i}+E \cdot T_{j}$$，而$$\tau$$是通过实验所得，使在验证集上获得最大的F1

- 在本次实验中并未使用TriviaQA data set
- 超参：batch size = 48，epochs = 2，learning rate = $$5 \times 10^{-5}$$



### 4.3 SWAG

- 全称The Situations With Adversarial Generations，用于常识推断任务，具体任务是给定一个sentence，然后需要在4个选择中选出最合适的答案
- 任务可建模为：每次有4个输入序列，每个输出是给定的sentence+4个可能的选择之一，最后得到C向量，再加一层全连接层，用sotfmax计算概率
- 超参：batch size = 16，epochs = 3，learning rate = $$2 \times 10^{-5}$$





# 5 BERT和其他模型的对比

- 实验进行了ELMo，OpenAI GPT和BERT之间的对比
- 首先介绍大致做法和结构：

> 1. BERT使用双向Transformer，OpenAI GPT使用LTR Transformer，而ELMo使用LTR和RTL的LSTM在最后的简单连结
> 2. BERT和OpenAI GPT使用fine-tuning approaches，而ELMo使用feature-based approach
>
> <img src="https://zlkqzimg-1310374208.cos.ap-chengdu.myqcloud.com/image-20220702121317344.png" alt="image-20220702121317344" style="zoom:80%;" />

- 另外，ELMo使用在最后将LTR和TRL简单的连结，有以下缺点：

> 1. 两倍的工作量
> 2. 对于有些任务是不直观的，如QA
> 3. BERT在每层都可以实现双向，而ELMo只会在最后连结
