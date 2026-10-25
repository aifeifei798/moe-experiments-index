# Q/O 整形：四篇最近的"近亲"（详细版）

> 这份是给自己看的资料整理，配合 `README.md` 第四节「研究假设：Q/O 曲线」使用。
> 用词按台账习惯：**整形 / EQ / 削峰 / 母带**。每节末尾的英文名只是检索用的。

**一句话结论**：这四篇合起来把一件事说清楚了——**只做线性增益不可能改变模型输出（数学等价），能改变输出的只有"削峰"这类非线性操作；而被削掉的恰恰是模型功能性的 bias 项。** 所以「困惑度变差、回复变好」不是异常，是这条路径的必然代价。

---

## 0. 为什么是这四篇

你的操作是：读 `q_proj` / `o_proj` 的输出幅值 → 超过某个 ceiling 的部分用 tanh 软拐点压回去 → 不改权重、不训练。

拆开看只有三个要素：**在哪里动手（q/o）**、**用什么算子（软拐点/增益）**、**什么时候做（训练时还是推理时）**。这四篇正好各占一角：

| 工作 | 动手位置 | 算子 | 时点 | 目的 |
| :--- | :--- | :--- | :--- | :--- |
| Logit soft-capping | attention logits | tanh 软拐点（全局固定 cap） | **训练时** | 训练稳定 |
| QK-norm | 点积前的 q、k | 归一化（等价于控温度） | **训练时** | 训练稳定 |
| SmoothQuant | 线性层输入/权重 | 逐通道线性增益（数学等价） | **推理时**（PTQ） | 省位宽 |
| Massive Activations | 残差流里的极值激活 | 只观察，不动手 | — | 解释为什么不能乱动 |

空缺的位置很清楚：**推理时动手 + 非线性削峰 + 目标是质量而不是位宽**——这四篇谁都没占。这就是你的位置。

---

## 1. Logit soft-capping（Gemma 2）

### 它做了什么

在每一个注意力层，以及最后的输出层，把 logits 过一个软限幅：

```
logits ← cap · tanh(logits / cap)
```

- 注意力层 `cap = 50.0`，最终输出层 `cap = 30.0`
- 效果：把 logits 压在 (−cap, +cap) 区间内，**不硬截断**，靠近边界时斜率平滑变小

### 动机

不是为了让模型更聪明，是为了**别在训练里炸**。大模型训练时 logits 会长到很夸张的值，softmax 之后变成近似 one-hot，梯度变得极端，训练不稳定。软限幅把动态范围压住，同时保留梯度（虽然在大值区梯度会变小）。

### 关键点（对你最有用的一条）

它和你的高拐点**是同一个算子**——都是 `tanh` 软拐点，都在压极值。区别只有两个：

1. **位置**：它在 softmax 之前的 logits 上；你在 `q/o` 的输出幅值上。数学上不完全一样（你是通过改 q/o 间接改 logits），但作用方向一致。
2. **时点**：它在训练时烧进架构，模型从一开始就在这个约束下学；你是训练完之后挂上去，模型是在**没有这个约束的条件下学出来的**。

### 和你的关系

**你这套本质上是"给一个已经训练好的模型补做 soft-cap"。** 文献里没有人这么干过——所有 soft-cap 都是训练时用的。这既是你的空位，也是你需要回答的问题：模型没在约束下训练，事后补上会怎样？

你现在观察到的"NLL 变差、回复变好"就是这个问题的第一个答案：**事后补约束会让模型偏离它被训练出来的分布，但可能把注意力分布掰回更合理的位置。**

### 可以立刻做的对照

- 同一个模型，把 cap 设成 50（对齐 Gemma 2 的注意力层数值）跑一遍，看是不是也会掉 NLL。如果会，说明你的现象不是你的算子独有的，是"事后削峰"的通性——这反而更好写。
- 反过来，如果 cap=50 不掉 NLL 而 cap=24 掉，说明存在一个"安全天花板"，那就是你要找的边界。

> 检索名：logit soft-capping / soft capping / Gemma 2 (arXiv 2408.00118)

---

## 2. QK-norm（Gemma 3 / Qwen 3 / ViT-22B / Chameleon）

### 它做了什么

在算 `q · k` 之前，先对 q 和 k 各做一次归一化（通常是 per-head 的 RMSNorm，带可学的缩放）：

```
q ← RMSNorm(q) · γ_q
k ← RMSNorm(k) · γ_k
logits = q · kᵀ / √d
```

### 动机

和 soft-capping 完全一样：**控 logit 尺度**。但它用的不是"压极值"，而是"从源头保证范数不失控"。因为 q、k 的模长被归一化了，点积的尺度就基本固定，logits 自然不会飙。

### 为什么值得单独看

Gemma 3 直接用它**替换**了 Gemma 2 的 soft-capping，并且官方说法是"更好的精度 + 更快的速度"。Qwen 3 也用了。这说明业界在这件事上的判断是：**从源头控尺度（归一化）优于事后压极值（削峰）**。

对你的含义：

- 如果你的目标是"长期稳定的模型整形"，归一化思路更干净，但它必须在训练时做（否则等于改了模型的计算图）
- 你的 EQ 是**逐块、逐通道、局部**的，QK-norm 是**全局统一**的。局部化正是你能做而它做不到的地方——某一层离群严重就只压那一层，而不是全模型一起归一化
- 一个很值得做的对照：**同一模型上，QK-norm 式归一化 vs 你的逐层 EQ**，谁的 ppl 代价更小、谁更能压住尾部。这个对照目前没人做过

> 检索名：QK-norm / qk-layernorm / Query-Key Normalization (arXiv 2010.04245) / ViT-22B / Chameleon / Gemma 3 (arXiv 2503.19786)

---

## 3. SmoothQuant（2023，ICML）

### 它做了什么

逐通道的线性缩放，把激活的量化难度迁移到权重上：

```
s_j = max(|X_j|)^α / max(|W_j|)^(1−α)
W' = W · diag(s)        # 权重乘上去
X' = X · diag(s)^(−1)   # 激活除回来
```

因为 `W' X' = W X`，**数学上完全等价**，浮点下输出一模一样。`α` 一般取 0.5，在"权重好量化"和"激活好量化"之间取平衡。

### 它解决了什么

激活里有极少数通道的值比其余大 ~100 倍，INT8 量化时这些通道把整个动态范围吃光，其他值全被量化成 0。缩放之后激活范围变均匀，权重的范围还能承受，于是 W8A8 可行。论文报的是 1.56× 加速、2× 显存下降，能把 530B 模型塞进单节点。

### 这条对你最重要，因为它给了你一个硬判据

**线性增益 = 等价变换 = 不改变输出。**

也就是说：如果你只做"逐通道乘一个系数"而**不做削峰**，理论上不可能让模型变好或变坏——输出是逐位相同的（浮点误差除外）。你观察到的所有变化，**只能来自非线性那部分（tanh 软拐点）**。

这条推论很值钱，因为它把实验空间砍掉一半：

| 做法 | 能不能改输出 | 能不能提升质量 |
| :--- | :--- | :--- |
| 纯线性增益（SmoothQuant 式） | 不能 | **不可能**（除非是和别的操作组合，或改变了数值精度） |
| 削峰 / 软拐点（你的做法） | 能 | 有可能 |
| 归一化（QK-norm 式） | 能（改了计算图） | 有可能，但要重新适应 |

所以别再花时间扫"纯增益"的系数了——它的数学上限就是"不变"。

> 检索名：SmoothQuant (arXiv 2211.10438) / per-channel smoothing / W8A8

---

## 4. Massive Activations（Sun et al., 2024）

### 它发现了什么

LLM 里有极少数激活值，比其余的大出**五个数量级**（1e5 倍）。它们的性质很特别：

1. **位置固定**：集中在特定维度上，不是随机出现的
2. **值几乎不随输入变**：不管 prompt 是什么，这些值都差不多
3. **作用是 bias**：论文的结论是它们充当"不可或缺的 bias 项"
4. **会把注意力吸过去**：它们导致注意力概率高度集中到对应的 token 上，形成 attention sink
5. **不能删**：去掉之后模型直接崩

### 为什么这条最扎心

因为它直接说明：**离群不是噪声，是功能。** "把曲线抹平模型就会更好"这个朴素直觉，在文献里恰好是反的——人家花了一整篇论文论证这些尖峰不能动。

这也解释了你自己的 sham 实验：**起作用的是 ceiling，不是"选得准"**。因为真正有效的是"把极端峰值压下去"这个动作本身，而不是"挑对了哪些块"。而压峰值，就是在动功能性 bias。

### 那为什么你还能观察到"回复变好"

一个自洽的解释是：**功能性 bias 和注意力塌缩是同一件事的两面。**

- 尖峰把注意力吸到自己身上（attention sink）→ 注意力分配变窄 → 长上下文里"看不见"中间的内容
- 压掉一部分尖峰 → 注意力摊开一点 → 指令跟随、长文一致性变好（你觉得"回复好了"）
- 但同时模型失去了它依赖的 bias → 下一 token 的似然下降（NLL 变差）

**这不是矛盾，是同一操作的两个后果。** 你在两者之间找一个甜点，本质上是在做"注意力分配 vs 似然"的权衡——和母带里"响度 vs 动态范围"完全同构。

### 对你的直接含义

- 甜点会很窄：只能压 **p99.9 以上**的峰，动了 p99 就开始切功能
- 不要指望"全面提升"：这是重新分配，不是增加能力
- 评估必须分两轴：注意力端（熵、长文命中）+ 似然端（NLL、ppl）

> 检索名：Massive Activations (arXiv 2402.17762) / attention sink / outlier features (LLM.int8, Dettmers et al. 2022)

---

## 5. 把四篇串成一条链

```
① SmoothQuant  → 线性增益是等价变换，改不了输出
                        ↓
② 要改输出，只能上非线性（削峰 / 软拐点）
                        ↓
③ 削峰动的是 Massive Activations —— 功能性 bias，删了会崩
                        ↓
④ 所以必然：注意力分布改善（回复变好）+ 似然下降（ppl 变差）
                        ↓
⑤ soft-capping / QK-norm 早就知道这事，所以选择在【训练时】做，
   让模型从一开始就适应这个约束
                        ↓
⑥ 空缺：没有人试过【训练后】补做，并且用【任务指标】而不是位宽/NLL 来评判
```

这条链就是你那份假设的骨架。如果哪天要写成文章，**第 ⑥ 步就是你的 claim**，前面五步全是别人已经证明过的支撑，不需要你重新论证。

---

## 6. 你的 claim 边界（四条，展开）

| # | 差分点 | 展开 | 为什么重要 |
| :--- | :--- | :--- | :--- |
| 1 | **时点** | 所有近亲都在训练时做；你在训练后挂，零训练、可撤、可 per-deployment | 这是唯一没人占的位置 |
| 2 | **目标** | 它们为位宽（SmoothQuant）或训练稳定（soft-cap/QK-norm）；你为任务/感知质量，并接受 NLL 变差 | 评判标准不同，不能拿 ppl 互相比 |
| 3 | **算子性质** | 等价缩放（改不了输出）vs 非等价削峰（真的改）——所以必须当"模型手术"评估，而不是"重参数化" | 决定实验设计和别人的质疑点 |
| 4 | **粒度** | 它们都是全局统一 cap / 全局归一化；你能逐层、逐块、逐通道，还能看输入动态给量 | 局部化 + 动态化是真正的新自由度 |

四条都成立，才叫新东西。少一条就退回"已有工作的组合"。

---

## 7. 别人会问的三个问题（先准备好答案）

**Q1：这不就是 soft-capping 吗？**
> 算子一样，时点不一样。soft-capping 是训练时烧进架构的约束，模型在约束下学；我是在训练完成的权重上事后补，模型没为这个约束优化过。所以问题从"加不加"变成了"补多少、补在哪几层"，这本身是个新问题。

**Q2：ppl 变差了，凭什么说模型变好？**
> 因为目标是任务表现，不是似然。这和音频里"压缩后响度上去、动态范围下来"是一回事——失真指标变差，感知质量变好。我给了两轴指标（注意力熵/长文命中 + 硬指标 exact match），不是只看一边。

**Q3：Massive Activations 那篇说了不能动尖峰，你怎么解释？**
> 我同意不能"抹平"，所以我只动 p99.9 以上的部分，并且测出来起作用的是 ceiling 本身（随机块 sham 对照，KL 0.0015 vs 0.092）。我的结论不是"离群有害"，而是"在哪个 ceiling 上、用什么指标换什么"。

---

## 8. 检索关键词（写给未来的自己）

```
logit soft-capping / soft capping
QK-norm / qk-layernorm / query-key normalization
attention entropy collapse
massive activations / attention sink
outlier features / activation outlier
SmoothQuant / per-channel smoothing / W8A8
Outlier Suppression+ / DuQuant
activation shaping / activation clipping
inference-time intervention / steering vectors
```

**提醒**：搜 "EQ for models" / "model mastering" 什么都搜不到，这个词只在台账里用。

---

## 9. 参考文献

| 简称 | 出处 |
| :--- | :--- |
| Gemma 2（soft-capping 来源） | Gemma Team, *Gemma 2: Improving Open Language Models at a Practical Size*, arXiv:2408.00118 |
| Gemma 3（用 QK-norm 替换 soft-cap） | Gemma Team, *Gemma 3 Technical Report*, arXiv:2503.19786 |
| QK-norm 早期工作 | Henry et al., *Query-Key Normalization for Transformers*, arXiv:2010.04245 |
| ViT-22B（qk-layernorm 大规模验证） | Dehghani et al., *Scaling Vision Transformers to 22 Billion Parameters*, arXiv:2302.05442 |
| SmoothQuant | Xiao et al., *SmoothQuant: Accurate and Efficient Post-Training Quantization for LLMs*, arXiv:2211.10438 |
| Massive Activations | Sun, Chen, Kolter, Liu, *Massive Activations in Large Language Models*, arXiv:2402.17762 |
| 离群特征（LLM.int8） | Dettmers et al., *LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale*, arXiv:2208.07339 |
| Outlier Suppression+ | Wei et al., *Outlier Suppression+: Accurate quantization of LLMs by equivalent and effective shifting and scaling*, arXiv:2306.12964 |
| 注意力熵坍缩 | Zhai et al., *Stabilizing Transformer Training by Preventing Attention Entropy Collapse*, arXiv:2303.06296 |

> 数字（cap=50/30、α=0.5、1.56×、1e5 倍）都是原论文里的值；如果要引用到对外文档，建议再回原文核一遍页码。
