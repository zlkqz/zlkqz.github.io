---
title: ZeRO
data: 2023-07-26
math: ture
---



# 1 三种并行方式

### 1.1 数据并行

- 数据并行分为同时有server（更新参数的主gpu）和worker的**朴素数据并行（DP）**和没有server只有worker的**分布式数据并行（DDP）**

##### 1.1.1 DP

- **数据并行流程如下：**

<img src="https://zlkqzimg-1310374208.cos.ap-chengdu.myqcloud.com/image-20230730141424734.png" alt="image-20230730141424734" style="zoom:40%;" />

> 1. 每张卡都有一份模型权重备份，并且每张卡都喂入一个不同的micro-batch，分别进行一次FWD和BWD，得到各自得梯度
> 2. 进行一次all reduce操作，将梯度push到server gpu上，然后在server gpu上进行参数的更新
> 3. server gpu搞完之后再把更新结果（更新后的模型参数和优化器参数）广播到其他worker gpu上

- 但是这种方法具有很大的**显存冗余**，**对于server的通信负担也很大，并且在server更新参数的时候，其他卡都在空转**

- **可以通过异步更新的方法减少空转，即：**在server更新参数的时候，其他worker直接拿还未更新的参数和下一批batch的数据，继续进行FWD和BWD，相当于变相翻倍了batch size

<img src="https://zlkqzimg-1310374208.cos.ap-chengdu.myqcloud.com/image-20230730145643681.png" alt="image-20230730145643681" style="zoom:50%;" />



##### 1.1.2 DDP

- DDP的主要优化点就是：**去除了server，将server上的通讯压力均衡转到各个worker上，减少了单节点的通信负担**

- DDP在通信时的传输策略和DP不同，是采用**环状通信算法Ring-AllReduce**

- 现在的传输目标如下，假设有4块GPU，每块GPU上的数据也对应被切成4份：

<img src="https://zlkqzimg-1310374208.cos.ap-chengdu.myqcloud.com/image-20230730151217839.png" alt="image-20230730151217839" style="zoom:50%;" />

- Ring-ALLReduce的实现分为：**Reduce-Scatter**和**All-Gather**

- 在**Reduce-Scatter**过程中，每张卡一次只传输自身1/4的数据：

<img src="https://zlkqzimg-1310374208.cos.ap-chengdu.myqcloud.com/image-20230730151658273.png" alt="image-20230730151658273" style="zoom:50%;" />

<img src="https://zlkqzimg-1310374208.cos.ap-chengdu.myqcloud.com/image-20230730151738179.png" alt="image-20230730151738179" style="zoom:50%;" />

<img src="https://zlkqzimg-1310374208.cos.ap-chengdu.myqcloud.com/image-20230730151810709.png" alt="image-20230730151810709" style="zoom:50%;" />

- 可以看到，经过三次环状传输之后，每张卡拥有了1/4的完整数据：

<img src="https://zlkqzimg-1310374208.cos.ap-chengdu.myqcloud.com/image-20230730151911922.png" alt="image-20230730151911922" style="zoom:50%;" />

- 然后进行**All-Gather**操作，使每张卡都拥有完整数据：

<img src="https://zlkqzimg-1310374208.cos.ap-chengdu.myqcloud.com/image-20230730152041754.png" alt="image-20230730152041754" style="zoom:50%;" />

<img src="https://zlkqzimg-1310374208.cos.ap-chengdu.myqcloud.com/image-20230730152050446.png" alt="image-20230730152050446" style="zoom:50%;" />

- 在**All-Gather**同样进行三轮环状传播，每张卡即可得到一样的结果：

<img src="https://zlkqzimg-1310374208.cos.ap-chengdu.myqcloud.com/image-20230730152146123.png" alt="image-20230730152146123" style="zoom:50%;" />

- **采用环状算法的DDP和朴素DP的通信量是相同的，但是通信效率更高，并且减少了server节点压力**
- **Ring-AllReduce十分常用，在ZeRO，Megatron-LM中，它将频繁地出现，是分布式训练系统中重要的算子。**



### 1.2 模型并行

- 如果模型的规模比较大，单个 GPU 的内存承载不下时，我们可以将模型网络结构进行拆分，将一个Tensor分成若干份，把每一份分配到不同的 GPU 中分别计算
- 代表方法是Megatron-LM



### 1.3 流水线并行

- 将不同的 layer 分配给指定 GPU 进行计算。相较于数据并行需要 GPU 之间的全局通信，流水线并行只需其之间点对点地通讯传递部分 activations，**这样的特性可以使流水并行对通讯带宽的需求降到更低**

<img src="https://zlkqzimg-1310374208.cos.ap-chengdu.myqcloud.com/image-20230730153614898.png" alt="image-20230730153614898" style="zoom:80%;" />

- 然而，**流水并行需要相对稳定的通讯频率来确保效率**，这导致在应用时需要手动进行网络分段，**并插入繁琐的通信原语**

- 代表方法是GPipe





# 2 ZeRO

### 2.1 CUDA显存占用

- CUDA显存占用如下：

<img src="https://zlkqzimg-1310374208.cos.ap-chengdu.myqcloud.com/v2-fcc24ce92b951ca8515114204cfa59cf_1440w.webp" alt="img" style="zoom:40%;" />

- 其中Model States包括**优化器参数、梯度、模型参数**，这也是ZeRO所优化的部分
- **activation**表示前向传播时留的缓存，会在反向传播时用到。但是不是必须的，可以使用**Gradient Checkpoint**来反向传播的时候现算，会慢一些
- 剩余的存储空间基本就是：**通信buffer、碎片、CUDA核占用**



### 2.2 ZeRO Stages

- **ZeRO一般都是结合混合精度训练使用，当然也可以不结合**
- ZeRO其实就是**对三种Model States进行划分，每张卡存储一部分参数，在要用的时候再通信传输**，具体分为三个等级：

> 1. **Stage 1：**只划分优化器参数**（注意这里的优化器参数还包括fp32的模型参数备份）**
> 2. **Stage 2：**划分优化器参数、梯度
> 3. **Stage 3：**划分优化器参数、梯度、模型参数

- 设：模型参数为$$\Psi$$，卡的张数为$$N_d$$，由于有各种优化器，所以直接设优化器状态所占显存为$$K\Psi$$
- 由于采用混合精度训练，**模型参数和梯度为fp16，所以分别占$$2\Psi$$**。优化器以Adam举例，优化器状态包括：**模型参数备份（fp32）、momentum（fp32）、variance（fp32）**，分别占$$4\Psi$$，所以$$K=12$$

- 采用三种stages的显存优化效果如下（暂不考虑activation所占空间）：

<img src="https://zlkqzimg-1310374208.cos.ap-chengdu.myqcloud.com/image-20230730161151812.png" alt="image-20230730161151812" style="zoom:50%;" />



### 2.3 具体流程

##### 2.3.1 常用通信方式及其通信量

- 通信量分析都**指单卡传出的通信量**，而忽略传入的通信量，因为传入传出通信量相同且可以同时进行
- 并且这里的通信量不是像上面一样，指具体的字节数，而是直接指传输的参数数量

- **Reduce-Scatter：单卡通信量$$\Psi$$**

<img src="https://zlkqzimg-1310374208.cos.ap-chengdu.myqcloud.com/image-20230730162556796.png" alt="image-20230730162556796" style="zoom:50%;" />

- **All-Gather：单卡通信量$$\Psi$$**

<img src="https://zlkqzimg-1310374208.cos.ap-chengdu.myqcloud.com/image-20230730162542367.png" alt="image-20230730162542367" style="zoom:50%;" />

- **All-Reduce：单卡通信量$$2 \Psi$$（All-Reduce一般都是直接指Ring-AllReduce，即Reduce-Scatter + All-Gather）**

<img src="https://zlkqzimg-1310374208.cos.ap-chengdu.myqcloud.com/image-20230730162527164.png" alt="image-20230730162527164" style="zoom:50%;" />

- **不带ZeRO的DDP的通信量为**$$2 \Psi$$，只需要对梯度做一次**All-Reduce**



##### 2.3.2 Stage 1流程

1. 每张卡分别FWD和BWD得到各自的梯度
2. 对梯度做一次**Reduce-Scatter**，每张卡得到自己所属部分的$$\frac{1}{N_d}$$优化器状态对应的那部分的reduce后的$$\frac{1}{N_d}$$梯度（图中蓝色部分），**产生单卡通信量$$\Psi$$**
3. **并且在梯度汇总完之后，不属于自己的那$$1 - \frac{1}{N_d}$$梯度可以直接丢弃（图书白色部分）。然后现在每张卡有且只拥有自己所对应的$$\frac{1}{N_d}$$优化器参数以及$$\frac{1}{N_d}$$梯度，所以可以更新自己所属的$$\frac{1}{N_d}$$参数**

<img src="https://zlkqzimg-1310374208.cos.ap-chengdu.myqcloud.com/image-20230730165502320.png" style="zoom:40%;" />

4. 此时，每块GPU上都有$$1 - \frac{1}{N_d}$$的参数没有完成更新。所以我们需要参数对做一次**All-Gather**，从别的GPU上把更新好的$$\frac{1}{N_d}$$的参数取回来。**产生单卡通信量$$\Psi$$**

- **每张卡总通信量$$2 \Psi$$**



##### 2.3.3 Stage 2流程

1. 正常FWD，然后在BWD时，每张卡在做完$$\frac{1}{N_d}$$参数的BWD后，得到的这部分梯度有两个去向：**在当前卡上，用于用于继续上一层的BWD；将这部分梯度传到另一张卡上进行梯度的reduce**
2. **这两个去向都结束之后，这部分梯度就可以删除了，每张卡的BWD产生通信量$$\Psi$$**
3. 现在每张卡都带有每张卡有且只拥有自己所对应的$$\frac{1}{N_d}$$优化器参数以及$$\frac{1}{N_d}$$梯度，所以剩余流程和stage1一样，**产生单卡通信量$$\Psi$$**

- **每张卡总通信量$$2 \Psi$$**



##### 2.2.4 Stage 3流程

1. 在FWD的过程中，需要哪部分参数，对应的卡就需要把这$$\frac{1}{N_d}$$参数广播出来，才能继续的前向传播，**这部分参数用完之后马上丢，FWD产生单卡通信量$$\Psi$$**
2. 然后BWD流程和stage 2差不多，但是在BWD时同样需要用到对应的$$\frac{1}{N_d}$$模型参数，所以每张卡需要传输参数和梯度，**BWD产生单卡通信量$$2\Psi$$**
3. **由于每张卡只维护$$\frac{1}{N_d}$$参数，所以最后不再需要对参数再做一次All-Gather，得到reduce后的$$\frac{1}{N_d}$$梯度后，直接更新自己的$$\frac{1}{N_d}$$参数即可**

- **每张卡总通信量$$3 \Psi$$**





# 3 ZeRO Offload

### 3.1 Offload 思想

- 在混合精度训练下，一次训练迭代大致分为：**FWD、BWD、fp32参数更新、fp32参数更新之后再转fp16**
- 设模型参数为$$M$$，Batch_size为$$B$$。**前两个过程的时间复杂度为$$O(MB)$$，后两个过程时间复杂度为$$O(M)$$**
- 而Offload的思想就是**将后两个时间复杂度较低的过程下放到CPU进行**

- 整体过程大致如下：

<img src="https://zlkqzimg-1310374208.cos.ap-chengdu.myqcloud.com/image-20230801151916236.png" alt="image-20230801151916236" style="zoom: 50%;" />

其中边的权重要么是2M（fp16），要么是4M（fp32）

- 另外，为了提高效率，可以将计算和CPU、GPU通信并行。GPU算完一部分梯度后，同时进行上一层的梯度计算和将梯度传输给CPU。同样的，CPU在参数更新的同时，可以将已经更新好的参数传给GPU



### 3.2 多卡场景

- 刚刚讲的是单卡场景，在多卡场景下，只需要每张卡都对应一个CPU进程即可，各算各的
- 每个CPU进程只更新属于自己的参数，然后将参数传给GPU，GPU再进行通信（Stage 1 && 2最后还会有All-Gather的通信）

