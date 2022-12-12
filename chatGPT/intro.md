# ChatGPT为何逆天？——OpenAI生成式语言模型的前世今生

> 作者：**王子涵**，中国人民大学大三本科生，现于加州大学伯克利分校交换。研究兴趣为自然语言处理、问答系统、对话系统等，个人主页zihanwang314.github.io

最近，OpenAI发布了其最新的语言模型ChatGPT。它可以与人对话，写作文，甚至编写代码

![图1](https://files.mdnice.com/user/25219/486ea410-5234-4951-81fc-16bde45c4609.png)


![图2](https://files.mdnice.com/user/25219/014c3602-928e-4992-b0a3-eac786d8a293.png)

此次ChatGPT的发布在互联网上激起了众多讨论，短短几天之内知乎相关问题已有超过五百万阅读，模型已经被大家玩出了“花”。例如，有用户使用它写情诗表白，帮自己写作业题，一位推特用户甚至让ChatGPT参加了SAT的考试，并取得了1020分的成绩。

ChatGPT有什么值得关注的地方？其背后的原理又是什么？本文将对其背后庞大的生成式自然语言处理模型做一个简要的介绍，帮助读者了解其基本原理。


## 1 模型原理介绍

### 1.1 Pre-trained Transformer

为了介绍当前最新的模型，首先需要介绍OpenAI的GPT(Generative Pre-trained Transformer)家族背后的Transformer语言模型。不同于我们熟知的循环神经网络(CNN)与卷积神经网络(RNN)，Transformer使用注意力(Attention)机制对文章进行建模。简单来说，Transformer由多个注意力块(Attention Block)堆叠而成，每个注意力块有一个多头注意力层(Multi-head Attention Layer)和一个前向连接层(Feed-Forward Layer)。建模过程首先通过词嵌入(Token Embedding)与位置嵌入(Position Embedding)相加为一句话的每个单词生成表征。在多头注意力层的一个头中，每一个单词的表征被转换成查询（Q）、键（K）、值（V）三个表征，一个单词通过查询与其他单词的依赖程度，将不同单词的值加权输出，传入下一层。用数学公式表示，即：

$$
Attention(Q,K,V) = {\rm softmax}(\frac{QK^T}{\sqrt{d_k}})V
$$

其中 $QK^T$ 代表加权相加后，不同单词对其他单词的依赖程度。为了加速收敛，这个值会除以一个系数 $\sqrt{d_k}$。为了使加权系数为1，采用softmax对依赖矩阵进行归一化。不同注意力头的输出将拼接起来作为多头注意力层的最终输出。而在前向连接层中，前一层的输出经过一个较大的中间层，通过激活函数引入非线性关系，最终经过输出层得到输出。

Transformer通过多头注意力机制，在每一层为不同的单词之间建模不同的依赖关系，因此对文本的建模能力有着极高的潜能。同时，Transformer还有Encoder，Decoder， Encoder-Decoder三种模式。简单而言，三种模型的区别如下表所示：


|<center>模型结构	|<center>输出	|<center>适用任务	|<center>相关模型|
| :--------- | :--: | -----------: | -----------: |
|<center>Encoder	|<center>单词序列的表征	|<center>完形填空，句子分类|<center>	BERT, ALBERT, RoBERTa  | 
| <center>Decoder	 | <center>单词序列后的单词 | 	<center>对话系统，提示模型 | 	<center>GPT，Transformer-XL  | 
| <center>Encoder-Decoder	 | <center>另一个单词序列 | 	<center>文本翻译，风格转换 | 	<center>T5, BART, LED

为了让Transformer发挥其威力，当前的模型体量相对于CNN,RNN模型都较为巨大。如研究人员使用较多的bert-base有1.1亿参数，而典型的大模型GPT-3有1750亿参数。如此大的模型，使用随机初始化必定会导致收敛时间极为漫长，同时最终收敛的位置很有可能已经在训练预料上过拟合。为了减慢收敛速度，提升模型泛化能力，当前的Transformer模型都需要经过预训练(pre-training)。预训练就是通过大量的自监督语料，设置一些较为通用的任务，让模型先通过这些任务进行训练，收敛后再通过原本的训练数据进行训练。常用的预训练任务有掩码语言模型(Masked Language Model), 句子重排序任务（Sentence Permutation）， 跨度边界目标（Span Boundary Objective）等，如下表所示：

| <center>任务名称 | <center>输入 | <center>输出
| :--------- | :--: | -----------: |
| <center>Masked Language Model | 	<center>A_CDE_G	 | <center>ABCDEFG|
| <center>Sentence Permutation | 	<center>CD.EF.G.AB | 	<center>AB.CD.EF.G|
| <center>Span Boundary  Objective	 | <center>A<b>B</b>___<b>F</b>G	| <center>ABCDEFG|

PS.“.”为不同句子的分隔符。粗体表示仅使用该表征进行预测。

预训练大幅提升了Transformer模型的通用性，在一套语料库上经过预训练的Transformer可以在数百种不同的任务上进行微调(fine-tune)达到较好的结果。研究人员还通过针对不同的需求，从训练语料、预训练任务、训练方式、模型结构上对基础模型进行了非常多的探索，也形成了如今Transformer模型百花齐放的局面。

### 1.2 GPT-3

GPT-3由OpenAI在2020年5月发表于arxiv，是当时最大的语言模型，需要800GB的储存空间，即使当前也只有企业级硬件才能够运行如此大的模型。 GPT-3的预训练语料库包罗万象，维基百科也仅占其全部训练语料的3%。然而，GPT-3的训练任务却十分简单：预测当前语言序列的下一个单词。这也是GPT-3能够通用于各种任务的原因——它见过非常多的语料，并已经从中挖掘出自然语言文本的模式信息。

然而，正是因为GPT-3是在通用语料上使用通用任务进行训练，因而GPT-3的推理过程十分具有技巧性，需要人工设计提示。例如对于完形填空任务，可以设计以下提示(prompt)：

“Alice was friends with Bob. Alice went to visit her friend ___. → Bob\
George bought some baseball equipment, a ball, a glove, and a ___ . →”

在通过以上提示猜测任务模式后，模型会预测“bat”等单词，完成完形填空的任务。事实上，GPT-3最令人惊奇的便是其在少样本甚至零样本下的能力——无需对模型再次进行训练，只需设计提示词便可充分利用大模型的潜力。

GPT-3发布的发布，再次证明了人工智能界的一个事实——模型的大小与任务表现成正比。此后掀起了一股大模型训练的潮流，如AI21发布Jurassic-1(1780亿)，微软与Nvidia推出MT-NLG（5300亿），谷歌推出PaLM（5400亿）等。然而，这些模型或在模型结构上做出调整，或在训练过程中引入新的计算模式加速训练，但整体思路与GPT-3大同小异，学界对下一代人工智能模型仍然充满期待。

### 1.3 InstructGPT

2021年末，OpenAI开始采用文本与代码等混合模式训练新的大语言模型，如可以用于代码完成的基础模型code-davinci-002，基于InstructGPT进行训练的text-davinci-002及其升级版text-davinci-003等。

我们重点介绍InstructGPT(图3)。该模型目标在于对齐大模型对于人类指令的回应，使其更加善于遵循用户意图。其训练通过三个阶段完成。其中第一阶段为监督学习，第二、第三阶段为人类反馈下的强化学习。


![图3](https://files.mdnice.com/user/25219/5ea8d63f-9013-40d1-8e62-76680c0d531d.png)

第一阶段——通过监督指令数据微调。由于GPT-3使用通用文本进行训练，其中的对话文本所占比例有限。为了使GPT模型具有遵循人类指令的能力，需要通过监督数据进行微调。该阶段从指令数据集中抽样，依靠标注人员给出对应的高质量答案，分别作为监督数据的输入与输出进行微调。

第二阶段——训练奖励模型。第一阶段的模型所依赖的监督数据是有限的，如果持续训练下去，模型最终会过度遵循数据中给出的模式。然而，聊天场景大多数情况下是没有标准模式的，因此一种更好的方法是鼓励模型学习到不同答案的质量差别，即为GPT训练一个奖励模型。实验人员通过从指令数据集中抽样，并使GPT生成不同的回答。此后，实验人员对GPT的回答质量进行排序，作为监督数据训练奖励模型。奖励模型可以判断GPT后续生成的回答质量，从而产生大量的监督信号，指导GPT后续的生成。

具体而言，GPT针对输入 $x$ 将一次生成 $K$ 个回复 $\{y_1, y_2, ...,y_K\}$，实验人员对K个回复进行排序，并可以从中抽取 $\binom {K} {2}$ 个二元组 $(y_w, y_l)$, 其中 $y_w$ 是比 $y_l$ 更好的回复。奖励模型 $r_θ$ 对 $K$ 个回复计算奖励，通过以下损失函数：

$$
loss(\theta)=  -{ {1} \over {\binom {K} {2}}}E_{(x,y_w,y_l) \sim D}[log(\sigma(r_{\theta}(x,y_w)-r_{\theta}(x,y_l)))]
$$


优化模型。具体而言，σ函数内部对每个二元组计算奖励差，通过该损失函数可以鼓励模型优化过程中 $y_w$ 输出的奖励值大于 $y_l$。

第三阶段——通过近端策略优化算法（PPO）训练GPT。PPO是一种强化学习算法，通过奖励模型给出的奖励，学习回复的最佳策略。为了理解PPO模型，我们先来回顾强化学习中的几个概念：状态(s), 动作(a), 奖励(r)，策略(π)，优势(A)。策略是一个当前状态下所采取动作的概率分布，而优势是一个相对概念，表示在当前状态下采取当前动作所获得的奖励与当前策略所能获得的奖励的期望之差，是动作、状态、策略的函数。在InstructGPT的语境中，状态即提示词X, 动作即模型的回复y, 策略π为语言模型，而r通过奖励模型得出。

PPO的基础是策略梯度(Policy Gradient), 其优化目标为:

$$
L = A^{\pi}(s, a)
$$

即，在当前策略的基础上优化模型，使得通过梯度下降产生的新策略总是能够比当前策略获取更多的奖励。

策略梯度的稳定性并不够好，而PPO为了使得策略梯度更加稳定，进行了一定改进，奖励函数为

$$
L(s,a,\theta_k,\theta)=min \left(\frac{\pi_\theta(a|s)}{\pi_{\theta_k}(a|s)},(1+\epsilon)\right)A^{\pi_{\theta_k}}(s,a)\tag{A>0}
$$

$$
L(s,a,\theta_k,\theta)=max \left(\frac{\pi_\theta(a|s)}{\pi_{\theta_k}(a|s)},(1-\epsilon)\right)A^{\pi_{\theta_k}}(s,a)\tag{A<0}
$$

其中 $\theta_k$ 是用于比较的旧策略。可以看到主要有两个改进，一是将当前策略与旧策略的概率相除，避免对旧策略下表现比较差的数据点产生太大的梯度；二是将概率相除的结果进行截断，同样避免了梯度过高模型跑偏的问题。


在InstructGPT中，由于我们无法直接获得A,因此将A替换成以下函数：

$$ 
{\rm objective}(\phi)=E_{(x,y)\sim D_{{\pi_\phi}^{RL}}}[r_\theta(x,y)-\beta log\left( {\pi_\phi}^{RL}(y|x) / \pi^{SFT}(y|x) \right)] + \\
 \gamma E_x \sim D_{pretrain}[log({\pi_\phi}^{RL}(x))]
$$

其中， ${π_φ}^{RL}$ 为通过强化学习优化的策略， $π^{SFT}$ 为微调得到的策略。第一项将A替换成我们可以通过奖励模型得到的r，同时引入KL惩罚减轻模型的过度优化。同时，第二项将预训练梯度混合到PPO梯度中，以修复公共NLP数据集上的性能回归。

InstructGPT通过迭代第二阶段与第三阶段，反复训练奖励模型与策略模型，从而通过人类反馈不断改进生成质量。实验结果表明，InstructGPT 模型在回复的真实性方面比 GPT-3 有很高的改进，毒性(toxicity)略有改善，但偏差(bias)改善较小。此外，与 GPT-3 的输出相比，打标签者明显更喜欢 InstructGPT 输出。


### 1.4 ChatGPT

正如GPT-3之于GPT, ChatGPT相对InstructGPT没有在模型架构上做优化，而是采取了一套新的数据的收集机制。我们可以在官方给出的示例中看到，在新的数据收集机制下，模型的毒性降低了很多，忠实性也有很大的提升。

![图4 ChatGPT与InstructGPT回复的对比(翻译)](https://files.mdnice.com/user/25219/9eaeb2d5-fa86-4fc9-965e-39b47cb4c00d.png)

目前，这套机制并没有具体披露，不过我们可以参考Deepmind Sparrow对话模型数据收集的相关标准，可见[链接](https://arxiv.org/pdf/2209.14375.pdf)。

## 2 总结与展望

作为GPT系列模型的最新成员，ChatGPT在原有的语言模型上引入了基于人类反馈的强化学习模型，通过更高质量的语料、奖励数据收集方法，使它能够更好地处理对话任务，例如， ChatGPT 可以回答后续问题、承认错误、挑战不正确的前提并拒绝不适当的请求。此外，它还支持更多类型的任务，如写作文和编写代码。

除了这些优点，ChatGPT也有一些劣势。首先，ChatGPT 暂不能通过上网搜集资料对回复进行增强，这导致其有时会写出看似合理但不正确或荒谬的答案，也无法回答时事相关的问题。其次，目前仍然是基于单纯的概率建模方式进行建模，因此模型无法解决复杂甚至是普通的逻辑问题，如小学数学题。此外，当前的ChatGPT仍然不够安全，通过用户的诱导依然可以生成对社会有害的内容。这些都是接下来的研究中需要关注的话题。

当然，ChatGPT只是一个新的开始，是人工智能与人类智能“对齐”的一次新尝试。我们期待人工智能领域更多的突破，能够使AI更加理解人类的思维方式，从而更好地与人类沟通。在未来，我们相信新的工作会在优化模型结构、增强学习能力、实现安全控制等方面取得更大的进步。

## 参考资料：

1. [ChatGPT website](https://openai.com/blog/chatgpt/)
2. [Training language models to follow instructions with human feedback.](https://arxiv.org/pdf/2203.02155.pdf)
3. [Attention is All You Need.](https://arxiv.org/pdf/1706.03762.pdf)
4. [Language Models are Few-Shot Learners.](https://arxiv.org/pdf/2005.14165.pdf)
5. [Berkeley Deep Learning Course CS182/282A](https://inst.eecs.berkeley.edu/~cs182/fa22/)


