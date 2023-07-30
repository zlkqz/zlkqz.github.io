---
title: EDA和AEDA
math: true
date: 2022-9-24
---



- 本节主要介绍NLP领域的两种简单数据扩充方法：**EDA和AEDA**
- 还有许多其他的数据扩充方法， 例如将文本进行back-translation，即将文本翻译一次又翻译回去，从而扩充文本，还可以通过各种深度学习模型进行扩充。**但是这些方法都太过"expensive"，而EDA和AEDA就相比之下比较简单，只需要在输入文本之前一定的预处理即可。**



# 1 EDA

### 1.1 EDA的基本方法

- **EDA的基本方法包括四种：**
> 1. **Synonym Replacement (SR，同义词替换)：**随机挑选n个词**（不能是停用词）**，然后将每个词随机替换成同义词
> 2. **Random Insertion (RI，随机插入)：**挑选随机词**（不能是停用词）**的随机同义词，插入随机位置，进行n次
> 3. **Random Swap (RS，随机交换)：**随机挑选两个词，交换位置，进行n次
> 4. **Random Deletion (RD，随机删除)：**使用概率p随机删除每个词

- EDA的做法是，**对输入的句子进行改变，但是尽量不改变其句意，也就是使句意和true label尽量对应**，所以使用同义词替换等方法来增加噪音，但不能增加过多。其中，对于长句子，相比于短句子，能吸收更多的噪音，更能保持true label
- 进行SR和RI时，不是选择随机词进行操作，而是使用同义词，**目的就是为了尽量不改变原始句意**
- **超参的选择：**

> 假设句子长度为$$l$$，则$$n=\alpha l$$，$$\alpha$$表明了多少比例的词语会被改变。并且对于RD，我们使用$$p=\alpha$$。对于每个句子，我们创造$$n_{aug}$$个扩充句子



### 1.2 EDA不同模型上的表现

- 可以看到，EDA在RNN和CNN上实现了准确率的提升，并且对于小数据集，提升更为明显

<img src="https://zlkqzimg-1310374208.cos.ap-chengdu.myqcloud.com/image-20220808144843149.png" alt="image-20220808144843149" style="zoom: 80%;" />



### 1.3 不同数据集大小对EDA的影响

- 作者对多个数据集进行了测试，并且在最后（图f）给出了在所有数据集上的平均结果，以探究不同大小的数据集对EDA效果的影响：

<img src="https://zlkqzimg-1310374208.cos.ap-chengdu.myqcloud.com/image-20220808145353524.png" alt="image-20220808145353524" style="zoom:90%;" />

- 在图f中，不使用EDA的最高准确率是88.3%，是在使用所有数据集时实现的。但是使用EDA时最高准确率为88.6%，**甚至只是用了一半的源数据**
- **总的来说，EDA对于小数据集的影响更大**



### 1.4 EDA是否会影响True Label

- **作者的实验步骤是：**对于一个pro-con分类任务（PC），先不应用EDA进行训练，然后在测试集上，进行数据扩充（每个源数据扩充九个数据），将源数据和扩充数据一起输入模型测试，将最后一个dense层得到的向量使用t-SNE表示，然后得到如下结果：

<img src="https://zlkqzimg-1310374208.cos.ap-chengdu.myqcloud.com/image-20220808150515348.png" alt="image-20220808150515348" style="zoom:80%;" />

- 可以看到**扩充数据的潜在语义空间是接近源数据的**，所以对于多数情况，EDA是不会改变true label的



### 1.5 消融实验

- EDA是四种扩充方法的结合，而对于这四种方法，作者通过每次分别只使用一次方法，来探究四种方法各自的贡献和效果。并且对不同的$$\alpha$$取值进行选取：

<img src="https://zlkqzimg-1310374208.cos.ap-chengdu.myqcloud.com/image-20220808151914033.png" alt="image-20220808151914033" style="zoom:90%;" />

- 四种操作都获得了一定的提升，其中：
> - **对于SR：**使用小的$$\alpha$$获得了提升，但是过大的$$\alpha$$反而降低了表现，推测原因为：过多的替换改变了原本的句意
> - **对于RI：**提升对于$$\alpha$$的改变不是特别敏感，更为稳定，推测原因为：原本的词和相对位置保留了下来
> - **对于RS：**在$$\alpha \le 0.2$$时获得较大提升，但在$$\alpha \ge 0.3$$时出现了下降，推测原因为：交换过多的词其实就等同于将整个句子词语的顺序重新排列一遍
> - **对于RD：**小$$\alpha$$有很大的提升，但是大的$$\alpha$$十分影响表现，推测原因为：删除过多的词使句子变得无法理解

- **通过实验，作者推荐通常取$$\alpha=0.1$$**



### 1.6 扩充几句最为合适

- 其实就是对超参$$n_{aug}$$的选择，实验结果如下：

<img src="https://zlkqzimg-1310374208.cos.ap-chengdu.myqcloud.com/image-20220808154726577.png" alt="image-20220808154726577" style="zoom:80%;" />

- 可以看到，对于小数据集，$$n_{aug}$$最好大一些，而大数据集则不需要那么多扩充数据
- 作者还给出了**推荐的超参：**

<img src="https://zlkqzimg-1310374208.cos.ap-chengdu.myqcloud.com/image-20220808155215524.png" alt="image-20220808155215524" style="zoom:90%;" />



### 1.7 结论

- 尽管EDA实现了一定的提升，尤其是在小数据集上，但是仍有一定的**限制**：

> 1. 通过实验可以发现，**EDA在数据充足时，提升的效果是十分有限的**，基本都是1%不到
> 2. 并且就算是使用小数据集，**在使用pre-trained model时，如BERT等，得到的提升也是十分微小的**

- EDA的**本质作用**可以总结为以下两点：

> 1. 产生了一定程度的噪音，来**阻止模型过拟合**
> 2. 通过SR和RI操作，可以产生新的词典，使模型可以**泛化在测试集中而不在训练集中的词**





# 2 AEDA

### 2.1 AEDA的基本方法

- 其实就是随机位置插入随机标点，**插入次数选择$$1 \sim \frac{1}{3}sentence\_length$$的随机数**，插入的标点符号为：**{".", ";", "?", ":", "!", ","}**，举个栗子：

<img src="https://zlkqzimg-1310374208.cos.ap-chengdu.myqcloud.com/image-20220809125655875.png" alt="image-20220809125655875" style="zoom:67%;" />

- 对比一下EDA，**EDA的交换操作改变了文本顺序，并且删除操作会造成信息的损失，从而造成对模型的"misleading"**。而AEDA则会保留文本的顺序和词语。作者还做了详细的实验进行验证和对比



### 2.2 EDA和AEDA的对比

- 作者分别在CNN和RNN上进行了实验，进行数据扩充时，每个源数据扩充了16个数据，实验结果如下：

<img src="https://zlkqzimg-1310374208.cos.ap-chengdu.myqcloud.com/image-20220808175040407.png" alt="image-20220808175040407" style="zoom:75%;" />

- 可以看到EDA尽在小数据集上有所提升，但是在大数据集上表现更差了。但是AEDA在所有数据集上都有提升，尤其是在小数据集上更为明显。
- 作者认为造成这种结果的原因是：EDA的替换和删除操作给模型增加了许多"misleading"的信息

> The reason why EDA does not perform well can be attributed to the operations such as deletion and substitution which insert more misleading information to the network as the number of augmentations grows. In contrast, AEDA keeps the original information in all augmentations  

- 此外，作者还通过不同的数据集，针对数据集大小展开了研究，结果如下：

![image-20220809125001403](https://zlkqzimg-1310374208.cos.ap-chengdu.myqcloud.com/image-20220809125001403.png)



### 2.3 扩充几句最为合适

- 作者还探究了每个源数据扩充几句数据最为合适，结果如下：

<img src="https://zlkqzimg-1310374208.cos.ap-chengdu.myqcloud.com/image-20220809125426286.png" alt="image-20220809125426286" style="zoom:70%;" />

- 作者并没有在论文中指出最合适的超参，但是个人觉得大多数时候**扩充一到两句**就够了



### 2.4 对于BERT的提升

- 作者对于BERT模型，进行了加EDA和AEDA的对比，每个源数据只扩充了一句，结果如下：

<img src="https://zlkqzimg-1310374208.cos.ap-chengdu.myqcloud.com/image-20220809132749401.png" alt="image-20220809132749401" style="zoom:80%;" />

- EDA反倒下降了表现（有可能是$$n_{aug}$$只有1），而AEDA实现了细微的提升（还是十分有限。。。）