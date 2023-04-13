---
layout: post
read_time: true
show_date: true
title: "Toolformer学习"
date: 2023-04-10
tags: [LLM]
category: nlp
description: "一种用来自监督构建带有api调用注释数据集的方法"
---

[Toolformer](https://arxiv.org/pdf/2302.04761.pdf)是meta ai在23年2月发表的论文，主要提出了一种新方法，可以教导大语言模型通过调用api来使用扩展工具。

这个方法首先通过自监督的方法构建了一个包含扩展工具调用的语料库，再结合扩展预料库和原始语料库没通过fine-tune的方式训练语言模型。

### 数据集构建

在论文中每个api调用由一个数组表示，$c=(a_c,i_c)$,$a_c$表示调用的api名称，$i_c$表示调用api所对应的输入。r表示api调用c的返回结果。
$$
e(c) = <API>a_c(i_c)</API> \\
e(c, r) = <API>a_c(i_c) \rightarrow r_i</API>
$$
其中`<API>`、`</API>`、`→`都是特殊的token，在实际使用中使用`[`代替`<API>`,`]`代替`</API>`，`->`代替`→`。文章中为了便于阅读，不进行这种替换。

接下来将未经调整的文本数据集$C=\{x^1,...,x^{|C|}\}$转换为带有api调用注释的数据集$C$,转换流程如下图所示。接下来会详细介绍下每个步骤。

![figure2](./assets/img/posts/20230410/toolformer_figure2.png)

#### API采样

对于每个api，都会写下提示词和示例$P(x)$，将这部分内容结合原始文本作为上下文输入语言模型，让模型基于此预测每个字符后面生成`[`的概率。

下面是调用QA系统api的示例，最后一个Input后面的$x=x_1,x_2,...,x_n$表示原始文本输入。

```latex
Your task is to add calls to a Question Answering API to a piece of text. The questions should help you get
information required to complete the text. You can call the API by writing "[QA(question)]" where "question" is the question you want to ask. Here are some examples of API
calls:
Input: Joe Biden was born in Scranton, Pennsylvania.

Output: Joe Biden was born in [QA("Where was Joe
Biden born?")] Scranton, [QA("In which state is
Scranton?")] Pennsylvania.

Input: Coca-Cola, or Coke, is a carbonated soft drink manufactured by the Coca-Cola Company.

Output: Coca-Cola, or [QA("What other name is Coca-Cola known by?")] Coke, is a carbonated soft drink
manufactured by [QA("Who manufactures Coca-Cola?")]the Coca-Cola Company.

Input: x

Output:

```

生成时从Output后面开始生成，每次添加x中的一个字符，获取语言模型预测下一个字符为`[`的概率，将x中每一个字符后面接`[`的概率记录下来，保留其中大于阈值$\tau_s$的位置，如果大于k个，则只保留k个候选位置。
$$
p_i = p_M(<API> | P(x), x_{1:i−1})
$$
得到所有候选位置后，从每一个候选位置开始调用语言模型，即将序列`P(x), x1, . . . , xi−1,[ ` 作为模型输入前缀，直到模型生成`]`作为终止。

**注意：**移除所有不生成`]`的示例。

#### API执行

将上面所有生成的形如`[QA("Where was Joe Biden born?")]`这类api调用进行执行，得到相应的调用返回结果$r_i$。

#### API过滤

将执行过api调用的例子生成如下三种形式的序列：

1. 包含返回结果。注意因为语言模型还未经过微调，此时将$c_i,r_i$插入原文中会因为没有和模型训练预料对齐导致文本连续性中断，所以选择直接使用$e(c,r)$序列。
2. 不包含返回结果。注意因为语言模型还未经过微调，此时将$c_i,\epsilon$插入原文中会因为没有和模型训练预料对齐导致文本连续性中断，所以选择直接使用$e(c,\epsilon)$序列。$\epsilon$表示空序列
3. 不包含接口调用。即原始序列$[x_1:x_i]$。


在过滤时首先使用下列公式计算上述序列的加权交叉熵损失：
$$
L_i(\textbf{z}) = −\sum_{j=i}^nw_{j-i}\cdot \log p_M(x_j|\textbf{z},x_{1:j-1})
$$
其中$w_i,i\in \mathbb{R}$是给定的损失权重。
>在原文没有提及权重给定方式，推测是根据文本重要性生成权重

在分别得到上述三个序列的权重L1,L2,L3后，保留$L1\geq max(L2,L3)+\tau$的生成示例。
### 模型微调
将通过上述操作构建的包含api调用注释的数据集替换到原始数据集中，对于多个api调用的文本，重复api调用数据集生成和替换数据集步骤。使用新的数据集微调语言模型，微调时使用标准语言模型目标，即自回归任务。
### 模型推理

在模型推理时整体和普通语言模型推理一致，只是在遇到字符$->$时会调用相应的接口，并将接口返回的内容拼接到原有的文本上，之后继续进行语言模型的推理。