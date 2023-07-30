---
title: BART总结
math: true
date: 2022-8-29
---



- 作者提出了一种seq2seq的模型，是一种去噪自编码器，大体的设计思路是：**使用随机的噪音破坏文本，然后使用该模型将模型恢复回来**，模型的**结构是BERT的Encoder+GPT的Decoder**，取名叫做BART（Bidirectional and Auto-Regressive Transformers）

- 由于BART是seq2seq的模型，所以相比于BERT，可以拿来做翻译任务。并且通过实验发现，**BART在文本生成和理解任务等方面是优于BERT的**

- 这种去噪自编码器的优点是：**在无监督预训练时，可以学得更加鲁棒的特征**



# 1 BART的结构

- BART的结构就是BERT的Encoder+GPT的Decoder，**对于Decoder，将原本的ReLu改为了GeLu。并且参数初始化改为服从$$N(0, 0.02)$$**
- base model分别有6个Encoder和Decoder，large model分别有12个
- 同等的规模，BART比BERT的参数量多10%
- **BERT和GPT和BART的对比：**

<img src="https://zlkqzimg-1310374208.cos.ap-chengdu.myqcloud.com/image-20220809212642716.png" alt="image-20220809212642716" style="zoom:90%;" />

BERT适合具有**双向表征和可并行化的优点**，但是由于其不是自回归的， 并且每个词是各自独立进行预测的，所以并**不适合文本生成领域**。而GPT由于其自回归性，可以用于文本生成。所以BART就将两者结合，结合两者有点，生成一个seq2seq模型，**使输入和输出不需要对齐**，可以用于文本生成、翻译等任务。





# 2 BART的Pre-training

- BART的预训练通过引入噪音破坏文本再恢复文本的方式进行学习，损失采用Decoder的输出和原文本的交叉熵
- **BART相较于其他去噪自编码器最大的优点就是：它可以应用任何文本破坏方式，而不是特定的方法**

> Unlike existing denoising autoencoders, which are tailored to specific noising schemes, BART allows us to apply any type of document corruption



### 2.1 BART中使用的破坏文本方式

- **Token Masking：**BERT的Mask策略

- **Token Deletion：**随机删除词

- **Text Infilling：**采样多个文本片段，每个文本片段长度服从$$\lambda = 3$$的泊松分布**（长度也可为0）**，每个文本片段用**单个**[MASK] token替换，替换成单个[MASK]能够迫使模型学习到一个片段中所缺失的token数量

- **Sentence Permutation：**按句号将文档分割成多个句子，然后随机打乱这些句子。
- **Document Rotation：**随机均匀地选择一个token，再旋转文档使文档以该token作为起始。该任务的目的是训练模型识别文档开头

- 举个栗子：

<img src="https://zlkqzimg-1310374208.cos.ap-chengdu.myqcloud.com/image-20220809214445327.png" alt="image-20220809214445327" style="zoom:67%;" />

- BART的一个关键优势是噪声的随意性，可以动用任何方式(包括改变长度)对原始文本进行破坏。**这种方式让模型学习过程中更多地考虑句子的整体长度，并对输入进行更大范围的转换，从而将BERT中MLM和NSP目标统一起来。**

> This approach generalizes the original word masking and next sentence prediction objectives in BERT by forcing the model to reason more about overall sentence length and make longer range transformations to the input  





# 3 BART的Fine-tuning

### 3.1 句子分类任务

- 方法类似于使用BERT中的[CLS]。**将相同的句子同时输入Encoder和Decoder，取Decoder最后一个时间步的输出**

<img src="https://zlkqzimg-1310374208.cos.ap-chengdu.myqcloud.com/image-20220809230327711.png" alt="image-20220809230327711" style="zoom:75%;" />

- 这种方法很像seq2seq模型翻译任务中的做法，以上图为例，区别在于翻译任务只在Decoder中输入A、B、C、D，而不输入E，然后期望输出A、B、C、D、E。而在此句子分类任务中，输入A、B、C、D、E，期望输出A、B、C、D、E、Label，只取最后一个时间步的Label，用作分类。



### 3.2 Token分类和序列生成

- **Token分类：**将整个文档输入encoder和decoder，每个token用其对应的最上方的decoder输出值用以分类

- **序列生成：**由于Decoder的自回归性，所以很适合序列生成，直接把数据输入进Encoder和Decoder（Decoder中输入的是label数据）即可

 

### 3.3 翻译任务

- 翻译任务有所不同，**在原本的Encoder前面又额外增加了一个随机初始化的Encoder**，结构如下：

<img src="https://zlkqzimg-1310374208.cos.ap-chengdu.myqcloud.com/image-20220809232721147.png" alt="image-20220809232721147" style="zoom: 85%;" />

- **新加的Encoder作用是：先将外语输入进去，然后通过该Encoder将其编码成带噪音的目标端语言，然后再通过BART降噪，作用和pre-training类似**

> These layers are trained to essentially translate the foreign language to noised English, by propagation through BART, thereby using BART as a pre-trained target-side language model

- 步骤：

> 1. 冻结BART的大部分参数，仅更新新增加的encoder、BART位置嵌入和BART每个encoder第一层的自注意力输入投影矩阵
> 2. 将所有模型参数进行少量迭代训练





# 4 对比试验

- 文章对比了不同预训练目标之间的影响，包括：

> 1. **Language Model：**与GPT类似，训练一个从左到右的Transformer语言模型。该模型相当于BART的decoder，只是没有交叉注意(cross-attention)
> 2. **Permuted Language Model：**该模型基于XLNet，采样1/6的token，并以自回归的随机顺序生成。为了与其他模型保持一致，这里没有引入相对位置编码和XLNet中的片段级的循环注意力机制
> 3. **Masked Language Model：**与BERT相同，15%的token用 [MASK] token替换，训练模型重建出这些被遮蔽掉的token
>
> 4. **Multitask Masked Language Model：**与 UniLM 一样，使用额外self-attention mask训练带遮蔽的语言模型。自注意力遮蔽按如下比例随机选择:1/6从左到右；1/6从右到左；1/3未遮蔽；剩余的1/3中前50%的未遮蔽，其余的从左到右遮蔽
> 5. **Masked Seq-to-Seq：**与MASS模型类似，遮蔽一个片段中50%的token，并训练一个序列到序列模型预测被遮蔽的tokens

- 实验结果如下：

<img src="https://zlkqzimg-1310374208.cos.ap-chengdu.myqcloud.com/image-20220809235547649.png" alt="image-20220809235547649" style="zoom: 67%;" />

- 通过实验对比，总结出如下结果：

> 1. 在不同的任务中，预训练方法的表现有显著差异。换句话说，预训练方法的有效性高度依赖于任务本身。比如，一个简单的语言模型在ELI5数据集上可以夺冠，但是在SQUAD上的结果却是最差的
> 2. 遮蔽Token至关重要。只使用旋转文档或句子组合的预训练目标则效果较差，效果较好的都是使用了token的删除或遮蔽作为预训练目标。此外，在生成任务上，删除token似乎比遮蔽token更胜一筹
> 3. 从左到右的预训练目标有助于文本生成任务。Masked Language Model和Permuted Language Model在文本生成任务上不如其他模型。而这两种模型在预训练阶段都没有用到从左到右的自回归语言模型
> 4. 对于SQuAD而言双向的encoder至关重要。因为上下文在分类决策中至关重要
> 5. 预训练目标并不是唯一重要的因素。这里的Permuted Language Model略逊于XLNet，其中一些差异可能是由于没有使用XLNet架构中的其他的改进，如相对位置编码和片段级的循环机制
> 6. Language Model在ELI5数据集上技压群雄，其困惑度远优于其他模型。这表明当输出仅受到输入的松散约束时，BART较为低效

- 同时实验还对比了几种文本破坏方法对任务的贡献到底有多少，发现**使用Text Infilling或Text Infilling + Sentence Shuffling得到的效果最好**





# 5 在各种下游任务上的表现

- 在此实验中，使用large规模的模型，预训练使用RoBerta的batch size=8000和steps=500000，以及使用BPE。预处理使用了text infilling和sentence permutation，并且mask掉了30%的token，重排所有句子  



### 5.1 自然语言理解任务
- 结果如下：

<img src="https://zlkqzimg-1310374208.cos.ap-chengdu.myqcloud.com/image-20220810000923604.png" alt="image-20220810000923604" style="zoom:80%;" />

- BART在自然语言理解任务上与其他先进模型不相上下。这表明**BART在生成任务上的进一步突破并不是以牺牲自然语言理解性能为代价**



### 5.2 自然语言生成任务

- 在微调时，使用了label smooth的交叉熵损失，平滑参数为0.1。并在生成时使用大小为5的束搜索

- 文本摘要任务结果：

<img src="https://zlkqzimg-1310374208.cos.ap-chengdu.myqcloud.com/image-20220810001908759.png" alt="image-20220810001908759" style="zoom:80%;" />

- 在这两个摘要任务上，BART 在所有度量指标上均优于之前的模型，但与人类的摘要结果相比仍然有差距

- 对话生成任务结果：

<img src="https://zlkqzimg-1310374208.cos.ap-chengdu.myqcloud.com/image-20220810002027643.png" alt="image-20220810002027643" style="zoom:80%;" />

- BART 在对话生成任务上的性能同样优于之前的模型
- 抽象QA任务结果：

<img src="https://zlkqzimg-1310374208.cos.ap-chengdu.myqcloud.com/image-20220810002302848.png" alt="image-20220810002302848" style="zoom:80%;" />



### 5.3 翻译任务

<img src="https://zlkqzimg-1310374208.cos.ap-chengdu.myqcloud.com/image-20220810002422747.png" alt="image-20220810002422747" style="zoom:80%;" />