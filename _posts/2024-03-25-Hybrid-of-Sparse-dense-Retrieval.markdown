---
title:  "Hybrid of Sparse-dense Retrieval"
date:   2024-03-25 11:08:15 +0800
categories:
  - machine learning
  - retrieval
tags:
  - dense retrieval
  - sparse retrieval
  - neural network
  - hybrid retrieval
  - search engine
---

# 稀疏检索和混合检索

# 一、概述

<aside>
💡 稀疏检索：通过刻画语义的稀疏文档表示，建立倒排索引来提升检索效率；
稠密检索：通过将输入信息(查询和文档)映射到独立的稠密空间，使用近似最近邻算法来做快速检索；
混合检索：将稀疏检索和稠密检索两种方式融合。

</aside>


# 二、稀疏索引（Sparse Retrieval）

传统的文本召回基于Term-based, 存在两个主要的提升点：

1. Query和Doc的Term失配问题
2. 语义依赖信息缺失

优化方向分类：

- Term re-weighting
- Term expantion
- Term re-weighting + Term expansion

## 2.1 Term re-weighting

### 2.1.1 DeepCT

<aside>
💡 引入预训练模型预估词权重

</aside>

1. 论文：**Context-Aware Sentence/Passage Term Importance Estimation For First Stage Retrieval**
2. 背景
    1. 为什么要做term re-weighting？因为传统基于统计方法算出的Term的表示方法在表征能力和语义表示方面效果欠佳。
3. 升级点
    1. 离线模型：使用Bert为每个Term学好表征上下文语义的向量来估计每个Term在文档中的重要性。
    2. 应用层面：并把它存入倒排索引里，在倒排拉链计算Query和Doc匹配分时，引入富含语义信息的Term重要性取代基于词频统计(TF-IDF、BM25)等方式计算的Term重要性。
4. 具体做法
    1. 使用Bert模型为每个term计算embedding表示
    2. 在embedding后添加一层MLP， 计算term重要性。
5. label
    1. query term recall作为对文本中term的真实权重值，公式如下：
    2. 分母是与文档D相关的查询Query数量，分子是文档D相关的查询Query且Query中包含term T的数量。
    3. 含义：搜索查询的问题能够反映出相关文档中的核心思想，因此一篇文档中的词，若在多个相关查询中出现，则说明该词应该比该文档中的其他词具有更高的权重。
    4. 同理query中的词权重
    5. 分母是与查询Query相关的文档Doc数量，分子是查询Query相关的文档Doc且Doc中包含term T的数量。
6. 总结
    - 优点：使用BERTTerm 重要性的计算
    - 缺点：没有解决Query和Doc的Term失配问题

## 2.2 term expansion

<aside>
💡 对Query和笔记侧做term expansion。主要解决Query和Doc的Term失配问题。

</aside>

### 2.2.1 Doc2Query

<aside>
💡 seq2seq模型根据笔记内容生成Query

</aside>

1. 论文：**Document Expansion by Query Prediction**
2. 具体做法：
    - 使用（query, relevant document）pair对来训练一个seq2seq模型，以Doc作为输入来预测Query.
    - 模型结果：sequence-to-sequence transformer model

### 2.2.2 DocT5Query

<aside>
💡 T5模型为笔记内容生成Query

</aside>

1. 论文：**From doc2query to docTTTTTquery**
2. 具体做法：
    - 基本流程于上述类似，模型结构上使用T5模型取代Transformer为笔记生成可能的查询Query
    - 补充到笔记后面，其它流程同倒排索引
    - 目标：减少Query和Doc的Term失配影响。

## 2.3 Term re-weighting & Term Expansion

<aside>
💡 使用端到端模型同时做term re-weighting和document expansion

</aside>

### 2.3. 1 SparTerm

<aside>
💡 利用BERT把Term re-weighting和term expansion融合到一个框架中同时做

</aside>

1. 论文：**SparTerm : Learning Term-based Sparse Representation for Fast Text Retrieval.**
2. 效果展示：传统方法 VS SparTerm
    - term weight效果比传统的基于词频统计的term weight效果好
    - 同时做了Term expansion, 并且为expansion出的term生成term weight.
3. SparTerm的模型结构
    1. 模型由两部分组成：
        1. importance Predictor(负责做Term re-weighting)
            1. step1. 先使用bert模型得到每个Term的Embedding
            2. step2. 通过矩阵映射成词表维度，得到当前term在词表中所有term上的一个概率分布
            3. step3: 把输入长度个分布融合成一个分布，可以取sum、取max等。
        2. Gating Controller(门控开关): 0/1开关，控制term是否要参与最终的表征。为了保证最终结果的稀疏性
            1. lexical-only: 出现在原始文档中的term, 把这些term的门控开关设置成1
            2. expansion-aware: 扩展term的门控开关也是1
            3. 做法
                1. Dense Term Gating层：代表每个Term是否参与最终稀疏表征的概率。
                2. Binary Term Gating层：通过Binarizer二值化器进行二值化。
                3. 把expansion出来的Term和原文中出现的Term拆开，二值化器域值不太一样。原文中出现的Term的概率值普遍再较高水位，而expansion出来的Term普遍在较低水位，所以需要拆开
                4. expansion部分
                5. 原文Term部分
                6. 最终门口开关的输出是这两个组合
    2. 这两个组件合起来工作，从而确保最终表征的稀疏性和灵活性。这样这个门控开关组件在原文中出现的Term和expansion出来的Term都能做的很好。两部分组件融合，按位点积
4. 损失函数
    - *训练样本：𝑅* = {(*𝑞*1*, 𝑝*1*,*+*, 𝑝*1*,*−)*, ...,* (*𝑞𝑁 , 𝑝𝑁 ,*+*, 𝑝𝑁 ,*−)}, 查询Query、一个文档正样本p+, 一个文档负样本p-, 过下模型分别求出query、doc正样本、doc负样本的稀疏表征。
    - 对比学习损失：要求query稀疏表征和doc正样本稀疏表征的相似度> query稀疏表征和doc负样本稀疏表征的相似度

### 2.3.2 SPLADE

<aside>
💡 SPLADE和SparTerm整体模型结构和思路一致，在SparTerm的基础上做了一些升级

</aside>

1. **论文：SPLADE: Sparse Lexical and Expansion Model for First Stage Ranking**
2. 升级点：
    1. 分布的聚合方式的升级
        1. 引用log函数做平滑，防止部分term的重要性占主导地位。
    2. 损失函数优化
        1. 从之前只有query自身的doc正样本和负样本，优化后多了batch内随机负采样得来的doc负样本。
    3. 结果稀疏性优化
        1. 把SparTerm里的门控开关抛弃，在SparTerm里是通过门控开关保证最终结果的稀疏性，在SPLADE只保留了Importance Predictor部分，通过正则化保证稀疏性.
        2. 添加 FLOPs 正则化参数，并在损失函数中添加正则化损失，为保证最终结果稀疏性。加了两个正则项分别对Query的稀疏性和Doc的稀疏性进行约束。

### 2.3.3 SPLADE v2

<aside>
💡 把历史上对稠密表征效果提升明显的方法引入到稀疏表征训练里，从而拿到效果收益
</aside>

1. 论文：**From Distillation to Hard Negative Sampling: Making Sparse Neural IR Models More Effective**
2. 升级点：
    1. 分布的聚合方式升级：从求加和SUM改成求最大值Max
    2. 蒸馏技术(DistilMSE)
        1. 对模型进行蒸馏，使用 cross-encoder 作为Teacher, 训练SPLADE模型。蒸馏时的损失函数为MarginMSE
    3. 难负样本构造
        1. 采用BM25算法挖掘的负样本（已有）
        2. batch内负采样（已有）
        3. 单塔挖掘到的hard-neg（新增）
    4. 预训练技术：使用cocondenser作为预训练模型，取代bert MLM预训练模型

### 2.3.4 SPADE

<aside>
💡 降低Query侧编码器复杂度，拿到在线infer时性能收益
</aside>

<aside>
💡 增加Doc-encoder复杂度，使用双编码器取代单编码器，为term weight和term expansion进行单独建模，并进行联合训练
</aside>

1. **论文：SpaDE: Improving Sparse Representations using a Dual Document Encoder for First-stage Retrieval**
2. 背景
    1. 笔记侧的infer可以离线做，但是由于query侧的infer要在线做，对RT和性能要求比较高。所以简化Query侧编码器结构。先让query侧和doc侧分开表征，通过之后的交互，再让这两向量变的匹配。
    2. 稀疏表征的提取从单编码器升级为双编码器。
3. 耗时：SPADE对比SPLADE在保证准确率的情况下，大大减少了耗时100ms vs 30ms。
4. 模型结构
5. 升级点
    1. 提出两个encoder的模型结构，query侧一个encoder, doc侧一个encoder。对比之前论文，降低Query-encoder复杂度，增加Doc-encoder复杂度。
        1. Query侧：通过切词，生产query表征，词表大小维度。
        2. Doc侧：由两部分组成，Term weight部分和Term expansion部分。
            1. Term Weight部分负责计算Doc中出现Term的重要性
            2. Term expansion部分负责扩展doc term, 并计算扩展出的term的重要性
    2. 提出term-reweighting和term expansion两个编码器协同训练方式
        1. 在warm up阶段，每个encoder独立训练
        2. 在finetune结果，拿每个encoder有较大损失的样本去训练另一个encoder.通过拿较难样本进行训练，这两encoder能产生互补。
    3. 稀疏性优化
        1. 为什么要做稀疏性优化？：在倒排检索时如果不保证稀疏性，会对效率和资源大打折扣。for倒排索引的查询效率
        2. 具体做法：
            1. doc侧剪枝：在term expansion编码的过程中，每个term仅做k个扩展。
            2. 语料库维度剪枝：删除一些功能向停用词的无意义term。“the”, "a"，从而保证语料库维度稀疏性。
6. 具体做法
    1. Bert模型输入：w是term weight部分输入，e是term expansion部分输入
    2. Bert模型输出：w是term weight部分输出，e是term expansion部分输出
    3. term weight编码器部分
        1. 利用全链接网络对bert输出计算term weight score.
        2. 把每个token映射为词表维度
        3. 使用max pooling把所有term聚合，从而生成doc侧term weight的表征
    4. Term expansion编码器部分
        1. 每个Term得到其词表空间的一个分布
        2. 进行两步，以确保稀疏性：1. 使用ReLu激活函数把负值置0
        3. 根据每个term的概率分布，从中取top k个概率最高的term, k是2-10之间的一个数字。其余位置设置为0.
        4. 最后使用max pooling，把d个分布融合成一个分布，得到doc侧expansion的表征
    5. 在线inference时，把把两个表征线性融合为1个，参数的值通常在[0.3, 0.5]
    6. 协同训练
        1. warm-up
            1. weight编码器的损失函数
            2. expansion编码器的损失函数
            3. 说明：expansion侧会比weight侧多了一个损失项。多的这个损失项是因为本文只对doc侧进行扩展，而没有对query侧进行扩展。所以希望doc侧扩展出的term要尽可能的能命中query侧的term.因此追加了新的损失项，强化doc侧和query侧在词命中上的能力
        2. Fine-tuning阶段：两个编码器各自把其损失大的样本挑出来去训练互相训练另一个编码器

# 三、混合检索（**Hybrid of Sparse-dense Retrieval**）

<aside>
💡 为什么要做混合检索？

稀疏检索，只要基于关键词匹配。在处理长文本和具有明确关键词的查询时表现良好，但在处理语义相似性和短文本时效果不佳。

稠密检索，通常基于深度学习模型，可以捕获文本的语义信息。在处理语义相似性和短文本有很好的效果，但可能无法很好地处理长文本和具有明确关键词的查询。

因此，通过混合检索，我们可以同时利用稀疏检索和稠密检索的优点，提供更准确和全面的搜索结果。

</aside>

## 3.1 CLEAR

<aside>
💡 稀疏检索和稠密检索进行联合训练，实现互补。

</aside>

1. CLEAR全称：Complementary Retrieval Model 互补召回模型
2. 论文：**Complement Lexical Retrieval Model with Semantic Residual Embeddings**
3. 模型结构：由两个模型组成Lexical Retrieval Model和 Embedding Retrieval Model
    1. Lexical Retrieval Model：用于捕捉token维度信息匹配, 使用**BM25**计算query和doc的匹配分
    2. Embedding Retrieval Model: 使用**sentence-bert**提取query和doc表征，使用两个向量点击计算匹配分
4. 训练方式
    1. 基于残差的学习框架( residual-based learning framework)，先用Sparse Retrieval进行检索将检索不好的查询再用Dense Retrieval进行增强。利用一个基于**sentence-bert**的dense retriever来学习sparse retrieval模型BM25的残差。
    2. Embedding Retrieval Model的负样本构造：Error-based Negative Sampling，取Lexical Retrieval Model错误召回的样本。被Lexical Retrieval Model召回但是不相关的笔记。这样向量召回模型就具备了识别字面匹配但是语义不匹配的鉴别能力。
    3. 损失函数优化：
        1. 文本召回有两个超参K和b需要调整。
        2. 向量召回损失函数里的margin值由静态改成动态。margin值根据Lexical Retrieval Model召回的错误率动态调整。lex代表Lexical Retrieval Model，向量召回模型损失函数里的动态margin由Lexical Retrieval Model的错误率决定。错误率越大，margin越小；错误率越小，margin越大。
        3. 向量召回模型的损失函数：动态margin的TripletHingeLoss.对TripletMarginLoss进行修改。
        4. 在Lexical Retrieval Model已经召回的样本上，margin值很小，向量召回模型进行很小的梯度参数更新。在Lexical Retrieval Model召回较差的样本上，margin值很大，向量召回模型进行很大的梯度参数更新。
5. 在线inference: 最终召回分融合了两个模型的召回分。

## 3.2 COIL

<aside>
💡 改变了倒排索引的构建方式，传统的倒排索引是(文本Term, 文本Term在doc中的统计信息)升级为（文本Term，Term在doc中的语义表征）

</aside>

1. 论文：**COIL: Revisit Exact Lexical Match in Information Retrieval with Contextualized Inverted List**

2. 常见四种模型类型：

    a类型：基于拼接序列的Cross-Attention，由于起高昂的计算代价，通常只能用于reranker

    b类型：deep retrieval中经典的双塔模型，存在上述的vocabulary mismatch问题

    c类型：ColBERT尝试融合lexical match与semantic match，但是要将document中的每一个词都和query词做交互，无论是索引构建和计算过程都需要提高代价

    d类型：COIL，相比于ColBERT，COIL只关注query与document中共同出现的词，因此索引构建的时候只需要在原有倒排索引上加上term embedding，运算过程也简便了很多

3. 模型结构：d类型：COIL，其中dot是针对[CLS]部分、max是针对普通token部分
    1. 使用Transformer对query的每个token和doc的提取token的embedding
    2. 使用MLP层把每个token的embedding维度降低。
    3. 基于向量相似度为每个token计算Query和Doc匹配分，比如Query中的bank会跟Doc中的每个Token计算SimScore, 最终取所有分的max代表Query和Doc在这个token下的匹配分。
    4. 对于Query和Doc Term失配问题：使用CLS位置分别代表整个Query和Doc的语义表征。这两个向量的相似度在代表语义上的相似度，从而从一定层面上缓解Query和Doc Term失配问题。
    5. 最终分是token维度匹配分和CLS维度匹配分之和
    6. 损失函数优化对比学习损失

4. 索引构建
    1. 笔记的Token表征和CLS表征离线算好，为每个Token构建语义倒排索引。为CLS构建倒排索引，文章中说也可以使用ANN（ approximate search index）去提高检索效率
    2. 离线索引构建
    3. 在线检索过程
        1. 先Term维度的Query向量和Doc向量计算相似度，在Matrix vector product环节计算，把这个分排序
        2. Term维度的分进行合并，得到最终的Query和Doc的召回分。

# 四、附录

https://zhuanlan.zhihu.com/p/613818089