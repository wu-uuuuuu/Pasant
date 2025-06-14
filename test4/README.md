
## 一、Transform

残差网络 `y = f(x) + x` 做归一化和标准化，保证数据训练的稳定性

###1. 总体结构

Transformer是由一堆Encoder和Decoder形成的

Encoder和Decoder均由多头注意力层和全连接前馈网络组成，网络的高层结构如下:

Encoder由N个编码器块(Encoder Block)串联组成，每个编码器块包含：

一个多头注意力(Multi-Head Attention)层；

一个前馈全连接神经网络(Feed Forward Neural Network)；

Decoder也由N个解码器块(Decoder Block)串联组成,每个解码器块包含:

一个多头注意力层；

一个对Encoder输出的多头注意力层；

一个前馈全连接神经网络；

###2. Transformer 输入

Transformer 中单词的输入表示 x 由 词嵌入(Embedding) 和 位置编码(Positional Encoding) 相加得到。

####1.1 词嵌入(Embedding)

Embedding 原理的详细介绍，请查阅博客：【Transformer系列】深入浅出理解Embedding（词嵌入）

Embedding 有很多种方式可以获取，例如可以采用 Word2Vec、Glove 等算法预训练得到，也可以在 Transformer 中训练得到。

####1.2 位置编码(Positional Encoding)

Positional Embedding 原理的详细介绍，请查阅博客：【Transformer系列】深入浅出理解Positional Encoding位置编码

Transformer 中除了 Embedding（词嵌入），还需要使用 Positional Encoding 表示单词出现在句子中的位置。因为 Transformer 不采用 RNN 的结构，而是使用全局信息，不能利用单词的顺序信息，而这部分信息对于 NLP 来说非常重要。所以 Transformer 中使用 Positional Encoding 保存单词在序列中的相对或绝对位置。

Attention中缺少一种解释输入序列中单词顺序的方法，它跟序列模型（RNN）还不一样。为了处理这个问题，Transformer对输入进行位置编码，以便在翻译中考虑单词在句子中的位置。具体来说，Transformer给encoder层和decoder层的输入添加了一个额外的向量Positional Encoding，维度和embedding的维度一样，这个向量采用了一种很独特的方法来让模型学习到这个值，这个向量能决定当前词的位置，或者说在一个句子中不同的词之间的距离。这个位置向量的具体计算方法有很多种，Transformer论文中使用一组正弦和余弦方程来实现，计算方法如下：

其中pos是指当前词在句子中的位置，i是指向量中每个值的index，可以看出，在偶数位置，使用正弦编码，在奇数位置，使用余弦编码。

最后把这个Positional Encoding与embedding的值相加，作为输入送到下一层。


###2. 编码器(Encoder)

####2.1 Encoder编码过程

首先，模型需要对输入的数据进行一个词嵌入(embedding)操作，也可以理解为类似w2c的操作，enmbedding结束之后，输入到encoder层，self-attention处理完数据后把数据送给前馈神经网络，前馈神经网络的计算可以并行，得到的输出会输入到下一个encoder。

####2.2 Encoder结构

Encoder 结构由 N = 6 \text{N} = 6N=6 个相同的 Encoder block 堆叠而成，Encoder block 由 Multi-Head Attention、Add&Norm、Feed Forward层组成。每个 Encoder block 的输入矩阵和输出矩阵维度是一样的，每一层主要有两个子层：

第一个子层是多头注意力机制（Multi-Head Attention）；

第二个是简单的位置全连接前馈网络（Positionwise Feed Forward）。

####2.3 Self-Attention

Self-Attention原理的详细介绍，请查阅博客：【Transformer系列】深入浅出理解Attention和Self-Attention机制

####2.4 Multi-Headed Attention

Multi-Headed Attention原理的详细介绍，请查阅博客：【Transformer系列】深入浅出理解Attention和Self-Attention机制

Multi-Headed Attention是在Self-Attention基础上改进的，也就是在产生q，k，v的时候，对q，k，v进行了切分，分别分成了num_heads份，对每一份分别进行self-attention的操作，最后再拼接起来，这样在一定程度上进行了参数隔离。

Multi-Headed Attention 不仅仅只初始化一组Q、K、V的矩阵，而是初始化多组，Transformer是使用了8组，所以最后得到的结果是8个矩阵。

Multi-head Attention的示意图如下：


####2.5 Add & Norm结构

在Transformer中，每一个子层（self-attetion，Feed Forward Neural Network）之后都会接一个残缺模块，并且有一个Layer normalization。

Add & Norm 层由 Add 和 Norm 两部分组成。这里的 Add 指 X + MultiHeadAttention(X)，是一种残差连接。Norm 指的是 Layer Normalization。

Add & Norm 层计算过程用数学公式可表达为：

其中，Add代表 Residual Connection 残差连接，是为了解决多层神经网络训练困难的问题。通过将前一层的信息无差的传递到下一层，可以有效的仅关注差异部分，这一方法在图像处理结果如ResNet等中常常用到。

Layer Normalization 是一种常用的神经网络归一化技术，可以使得模型训练更加稳定，收敛更快。

Layer Normalization和 Batch Normalization 有相同的作用，都是为了使输入的样本均值为零，方差为1。

Layer Normalization对每个样本在特征维度上进行归一化，减少了不同特征之间的依赖关系，提高了模型的泛化能力，其原理可参考论文：Layer Normalization。

####2.6 Feed Forward结构

前馈神经网络( Feed Forward Neural Network，简称FFN)，其本质是一个两层的全连接层，第一层的激活函数为 Relu，第二层不使用激活函数，计算过程用数学公式可表达为：

除了使用两个全连接层来完成线性变换，另外一种方式是使用 kernal_size = 1 的两个 1 × 1 1\times 11×1 卷积层，输入输出维度不变，都是 512，中间维度是 2048。

Feed Forward没法输入 8 个矩阵，这该怎么办呢？所以我们需要一种方式，把 8 个矩阵降为 1 个。首先，我们把 8 个矩阵连在一起，这样会得到一个大矩阵，再随机初始化一个权重矩阵，并与这个组合好的大矩阵相乘，得到一个最终的矩阵。

###3. 编码器(Decoder)

根据上面的总体结构图可以看出，Decoder的结构与Encoder结构大同小异，先添加一个 Positional Encoding 位置向量，再接一个masked mutil-head attetion，这里的mask是Transformer一个很关键的技术，本章节对其进行详细介绍。其余的层结构与Encoder一样，请参考Encoder层结构。

####3.1 Decoder结构

####3.2 Masked Multi-Head Attention的概念

mask 表示掩码，它对某些值进行掩盖，使其在参数更新时不产生效果。由于使用了 Masked Multi-Head Attention，所以每个位置的词只能看到前面词的状态，不会“看见”后面的词，所以 Masked Multi-Head Attention 是一个单向的Self-Attention结构。Transformer预测第T个时刻的输出，不能看到T时刻之后的那些输入，从而保证训练和预测一致。

####3.3 mask的分类

Transformer 模型里面涉及两种 mask，分别是 padding mask 和 sequence mask。其中，padding mask 在所有的 scaled dot-product attention 里面都需要用到，而 sequence mask 只有在 decoder 的 self-attention 里面用到。

通过 query 和 key 的相似性程度来确定 value 的权重分布的方法，被称为 scaled dot-product attention。

对于 decoder 的 self-attention，里面使用到的 scaled dot-product attention，同时需要 padding mask 和 sequence mask 作为 attn_mask，具体实现就是两个mask相加作为 attn_mask。

其他情况，attn_mask 一律等于 padding mask。

####3.4 padding mask

什么是 padding mask 呢？因为每个批次输入序列长度是不一样的，也就是说，我们要对输入序列进行对齐。具体来说，就是给较短的序列后面填充 0。如果输入的序列太长，则是截取左边的内容，把多余的直接舍弃。对于太长的输入序列，这些填充的位置，是没什么意义的，attention机制不应该把注意力放在这些位置上，所以我们需要进行一些处理。

具体的做法是，把这些位置的值加上一个非常大的负数（负无穷），这样经过 softmax，这些位置的概率就会接近0。

padding mask 实际上是一个张量，每个值都是一个Boolean，值为 false 的地方就是我们要进行处理的地方。

####3.5 Sequence mask

sequence mask 是为了使得 decoder 不能看见未来的信息。也就是对于一个序列，在 time_step 为 t 的时刻，解码输出应该只能依赖于 t 时刻之前的输出，而不能依赖 t 之后的输出。因此需要想一个办法，把 t 之后的信息给隐藏起来。

那么具体怎么做呢？也很简单：产生一个上三角矩阵，上三角的值全为0。把这个矩阵作用在每一个序列上，就可以达到我们的目的。

####3.6 Output层

当decoder层全部执行完毕后，怎么把得到的向量映射为我们需要的词呢，很简单，只需要在结尾再添加一个全连接层和 softmax层。假如词典有1w个词，那最终softmax会输入1w个词的概率，概率值最大的对应的词就是最终的结果。

## 二、训练自己的数据集
### 1.数据增强
```
train_data = ImageTxtDataset(r"D:\Desktop\tcl\dataset\train.txt",
                             r"D:\Desktop\tcl\dataset\image2\train",
                             transforms.Compose([transforms.Resize(256),
                                                 transforms.RandomHorizontalFlip(),
                                                 transforms.ToTensor(),
                                                 transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                      std=[0.229, 0.224, 0.225])
                                                 ])
                             )
```
### 2.调用GPU去训练

把我们的模型，数据，标签，使用” .cuda() “去推到GPU上
```
        if torch.cuda.is_available():
            imgs = imgs.cuda()
            targets = targets.cuda()`
#查看机器能否使用
gpu`import torch`, `torch.cuda.is_available()
```
