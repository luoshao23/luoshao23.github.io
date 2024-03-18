---
title:  "Some metrics for machine learning"
date:   2024-03-18 16:30:00 +0800
categories:
  - machine learning
tags:
  - metrics
---
(持续更新)

## [Perplexity(困惑度)](https://huggingface.co/docs/transformers/en/perplexity)


PPL用以衡量语言模型好坏。语言模型越好，在给定数据集上的PPL越小。

$$
PPL(X)=exp\{-\frac{1}{t}\sum_i^t{log(p_{\theta}(x_i|x_{<i}))}\}
$$

Pro：快速；Con：粒度影响结果

神经网络中的PPL计算往往不直接计算而是使用cross entropy loss

$$
PPL = 2^J, \\ J = -\frac{1}{T} \sum_{t=1}^{T} \sum_{j=1}^{|V|} y_{t,j} log (\hat{y}_{t,j})
$$

## BLEU（Bilingual Evaluation Understudy）

https://en.wikipedia.org/wiki/BLEU

https://zhuanlan.zhihu.com/p/350596071

模型输出的precision，BP为短语惩罚项，Pn为n-gram的精确度

$$
BLEU=BP*exp(\frac{1}{n}\sum_{i=1}^N P_n)
$$

## [ROUGE](https://blog.csdn.net/qq_39610915/article/details/117078443)

NLP中用于衡量自动摘要和机器翻译质量的指标。指标将自动生成结果和参考结果进行比较计算，阈值为0-1，越大说明越相似。

ROUGE = count(match)/total_num

ROUGE-N：n-gram下相同的个数的占比

ROUGE-L：

$$
R_{lcs}=\frac{LCS(X, Y)}{m}, \\ P_{lcs}=\frac{LCS(X, Y)}{n}, \\ F_{lcs}=\frac{(1+\beta^2)R_{lcs}P_{lcs}}{R_{lcs}+\beta^2 P_{lcs}}
$$

## Inception score (IS)

https://en.wikipedia.org/wiki/Inception_score

https://juejin.cn/post/7156610588137750564

https://zhuanlan.zhihu.com/p/646149258

用于判断生成模型的效果好坏（Fidelity and diversity），IS越大越好

- fidelity：模型预测类目标签`p(y|x)`的熵越小越好
- diversity：预测类目的分布`p(y)`越均匀越好，熵越大越好

$$IS(p_{gen}, p_{dis}):=exp(E_{x \sim p_{gen}}[D_{KL}(p_{dis}(y|x)||\int{p_{dis}(y|x)p_{gen}(x)dx})])$$


存在问题：只使用了预测模型结果，没有与ground truth对比。

互信息view: `I(y;x) = H(y) - H(y|x)`

## FID score

来源于Fréchet distance，评估图像生成模型的质量。衡量两个图像集隐式特征分布之间的差异，越小越相似

def：[https://en.wikipedia.org/wiki/Fréchet_inception_distance](https://en.wikipedia.org/wiki/Fr%C3%A9chet_inception_distance)

具体实现：https://github.com/bioinf-jku/TTUR/blob/master/fid.py#L109

The Fréchet distance between two multivariate Gaussians $$X_1 \sim N(\mu_1, C_1)$$
and $$X_2 \sim N(\mu_2, C_2)$$ is

$$
d^2 = ||\mu_1 - \mu_2||^2 + Tr(C_1 + C_2 - 2(C_1 C_2)^{1/2}).
$$

缺点：

- 需要使用特征抽取器
- 同样的pipeline
- 不敏感

## CLIP score

用于衡量模型生成的图片和对应的文字说明之间的相似度，越大越相似

论文：https://arxiv.org/abs/2104.08718

解释和实现：[https://unimatrixz.com/blog/latent-space-clip-score/#:~:text=CLIP Score is a widely,vision and language understanding tasks](https://unimatrixz.com/blog/latent-space-clip-score/#:~:text=CLIP%20Score%20is%20a%20widely,vision%20and%20language%20understanding%20tasks).

具体流程：使用CLIP模型，分别计算图片和文字的embedding，之后计算cosine相似度

## [Peak signal-to-noise ratio (PSNR)](https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio)

衡量原始图像和生成图像的之前的损失，越大越好

给定一张m*n单色图片，

$$PSNR=10\cdot log_{10}(\frac{max\{I\}^2}{MSE}) = 20\cdot log_{10}(max\{I\}) - 10 \cdot log_{10}(MSE)$$

其中max{I}是原始图片中最大的像素值，$$MSE=\sum[I(i, j) - K(i,j)]^2$$

## [Jensen](https://en.wikipedia.org/wiki/Johan_Jensen_(mathematician))–[Shannon](https://en.wikipedia.org/wiki/Claude_Shannon) divergence

测量两个概率分布之间的相似性，基于KL散度，[0, 1]，0表示完全一致

$$
JSD(P||Q) = \frac{1}{2}KL(P||M)+\frac{1}{2}KL(Q||M),\ where\ M=\frac{1}{2}(P+Q)
$$

优点：对称性、有界性、平滑性