# MoE Experiments Index

> 个人实验台账，不是论文，也不是导航页。**每个仓库 = 一次受控变量实验**，历史全部保留、不做合并。
> 这里只记三件事：这块改了什么变量、跑出了什么数、现在什么状态。
> 半年后要整合时，从这张表往下点，不用翻 40 多个 README。

状态图例：✅ 已验证（有可复现数字） · 🟢 主线候选 · 🟡 进行中 / 探索 · ⬜ 只有产物 · ⬛ 已被取代

---

## 一、Big-Little → Myriad 线（消费级显存墙）

核心问题：专家体积换容量必然撞显存，能不能让专家住在主机内存里、按 token 流式搬运，且不掉吞吐。

| 仓库 | 基线 | 改的变量 | 关键结果 | 状态 |
| :--- | :--- | :--- | :--- | :--- |
| [BigLittle-MoE](https://github.com/aifeifei798/BigLittle-MoE) | Qwen3-0.6B | 冻结大核 + 896 个 rank-16 LoRA（32×28 层）放主机 RAM 流式；路由头只占 1.75 MB 显存 | 专家池 bf16 共 56 MB（~64 KB/个）；Top-1 **36.5 t/s**、Top-8 22.4 t/s；单专家搬运 0.214 ms（p50）；v1.0.1 修正了旧版把吞吐算成 `45/elapsed` 的错 | ✅ 已验证（基线块） |
| [DualBigLittle-MoE](https://github.com/aifeifei798/DualBigLittle-MoE) | 上一块 | 双核：Arts 冻结原版 MLP + STEM 克隆核低学习率微调，896 LoRA 压到 54 MiB | 验证集 PPL：Code 7.58→**6.80**、Math 6.27→**5.63**、Arts 8.90→**8.20**（均 −9.5%）；896/896 专家非零、0/28 层饿死 | ✅ 已验证（v5_auxfix） |
| [Myriad-MoE](https://github.com/aifeifei798/Myriad-MoE) | DualBigLittle | 专家规模推到 20,000，pinned RAM 1.25 GB | README 空，仅剩一句 "Zero PCIe Choke" | ⬛ 已被 25K 取代 |
| [Myriad-MoE-25K-Micro-Experts](https://github.com/aifeifei798/Myriad-MoE-25K-Micro-Experts) | 同上 | 28 层 × 20 簇 × 45 个 LoRA = **25,200 专家**，1.54 GB pinned RAM，热插拔 | host 流式 **24.9 t/s** vs 显存常驻 23.8 t/s（**0% 吞吐损失**）；换卡带 44 ms；训练靠低秩张量收缩省 97.6% 显存 → **10 GB / 9.8 samples/s**；领域反转 Arts 79.1% vs STEM 20.9% | 🟢 主线候选（最完整） |
| [Myriad-MoE-E2B](https://github.com/aifeifei798/Myriad-MoE-E2B) | 25K | 换 E2B 底座 | 仓里只有一个 HF 权重链接 | ⬜ 仅权重 |
| [Gemma-4-12B Myriad 变体](https://github.com/aifeifei798/gemma-4-12B-it-qat-q4_0-unquantized-MoE) | Gemma-4-12B（冻结） | 只在 18–29 层嫁接 16 clan × 16 LITTLE，适配器 405.1M，true Top-K 稀疏 | v3（28k 数据、5205 步）：macro Top-1 **66.2%**（随机 12.5%）、clan Top-1 **45.6%**（随机 6.25%）、val LM **1.35**；峰值 ~13 GB，24 GB 卡可跑 | ✅ 已验证（v3） |
| [myriad-qwen3.5-0.8b](https://github.com/aifeifei798/myriad-qwen3.5-0.8b) | Qwen3.5-0.8B | 分类学换血：8 Macro × 16 Clan × 256 Micro + 前额叶意图投影（LoRA r=64） | 推理显存 **< 1.8 GB**，未选专家严格 ΔW≡0 | 🟡 探索中（新分类学） |

## 二、Heretic-Scalpel 线（三档适配器）

核心问题：把「大核锚定 + 稀疏小专家」做成可复用的微调范式，而不是一次性脚本。

| 仓库 | 基线 | 改的变量 | 关键结果 | 状态 |
| :--- | :--- | :--- | :--- | :--- |
| [Heretic-Scalpel-TriTier-E2B](https://github.com/aifeifei798/Heretic-Scalpel-TriTier-E2B) | Gemma 4 E2B | 双路由器 + dense STEM 核（r=64）+ 32 个 rank-16 微专家（96 KB/个） | 峰值 **17.4 GB**、**9.1 samples/s**（5090 D）；第 50 步 LM 稳定在 **~1.17** 且路由分化成型 | ✅ 已验证（单次训练曲线） |
| [Heretic-Scalpel-Swarm-Forge](https://github.com/aifeifei798/Heretic-Scalpel-Swarm-Forge) | TriTier-E2B（底座未公开） | 抽成框架：`forge_core` 纯 ML 库 + FastAPI + React | 修掉 8 个实质 bug（因果 LM 错位、N<32 路由越界、均衡损失梯度恒 0、251K 词表 logits 尖峰、导出切源码、数据集唯一率造假…）；200 步 / 6400 条：held-out LM **2.673 → 2.281** | 🟡 工具链进行中 |

## 三、从零复现线（架构对照）

核心问题：把前沿结构的每一个部件自己写一遍，验证「哪一部分真的在起作用」。

| 仓库 | 基线 | 改的变量 | 关键结果 | 状态 |
| :--- | :--- | :--- | :--- | :--- |
| [DeepSeek-Pro-Mini](https://github.com/aifeifei798/DeepSeek-Pro-Mini) | 无（从零） | MLA 隐空间注意力 + 8 专家选 2 + 1 共享 + sqrtsoftplus 门控 + 限幅 SwiGLU | 153M 总参 / **60M 激活** / 3.5 GB；预训练 **52k tokens/s**、245M token、90 万条 SFT；终端实时渲染 8 层专家放电 | ✅ 已验证（端到端） |
| [Mini-Architecture](https://github.com/aifeifei798/Mini-Architecture) | 无 | 单文件 `model_v41.py` 最小 V4.1 + engram | v3：预训练 val 5.369 → SFT 5.009；**稀疏度-质量单调**（4%→full 只差 0.30 loss，top-8 用 1/12 可见 token 拿到 90% 质量）；engram 用 3% 参数换 0.05 loss | ✅ 已验证（对照最规范） |

## 四、支撑件

| 仓库 | 作用 | 状态 |
| :--- | :--- | :--- |
| [myriad-moe-console](https://github.com/aifeifei798/myriad-moe-console) | 25K 模型的 Web 前端：流式对话 + 神经透视盘（簇热力、隔离、单层狙击、弹性 Top-K、卡带热插拔） | 🟡 工具（需自备后端） |

---

## 五、可以当事实用的结论（跨仓沉淀）

1. **专家住主机 RAM 不损吞吐**：每层搬运 ~0.13 ms，被 ~40 ms 的 forward 完全掩盖；host 24.9 vs device 23.8 t/s。
2. **Top-K 的 K 是成本不是收益**：Top-1 36.5 > Top-8 22.4 t/s，路由开销随 K 线性增长。Top-8 的用途是验证协同路由，不是提速。
3. **辅助损失必须逐层平均**：被 `/num_layers` 稀释 28 倍后，路由命中率 34~46%（等同随机猜），修复后 82~98%。看 loss 曲线看不出来，要看命中率。
4. **路由修好会牺牲文科保真**：Arts PPL 5.49 → 8.20。这是架构取舍，不是 bug；要同时保住两头得另找约束。
5. **稠密前向 + 软门控能让全部专家吃到梯度**：896/896 非零、0/28 层饿死，靠 `test_every_expert_receives_gradient` 锁住。
6. **权重差分不量化就不省体积**：bf16 差分仍是 504 MiB（Δ 只占 0.11%~0.21% 但不稀疏）；int8 差分 −45%，PPL 变化在小数第 4 位。
7. **单卡训 25,200 专家是可行的**：低秩张量收缩省 97.6% 训练显存 → 10 GB、9.8 samples/s。
8. **因果 LM 错位不报错但致命**：loss 17~19，而 `ln(262144)=12.5`，比随机初始化还差。必须有断言，否则训练照跑。
9. **251K 词表必须 chunk logits**：`B=4, T=256` 一份 bf16 logits 就 0.5 GiB，`.float()` 再翻倍。
10. **能烘回 stock 架构**：dense 版余弦 0.9935、MoE 版 5.62B 余弦 0.9988，可以不依赖 `trust_remote_code` 交付。

## 六、评测口径（这些数字不能混用）

- **采样方式**：域内分层等距 vs 取前 n 条，同一权重 Arts PPL 分别是 5.49 与 12.46。
- **数据切分**：验证集 vs 训练语料切片，绝对值不可比，只能看相对差。
- **吞吐**：必须按实际生成 token 计；旧版用固定 `45/elapsed` 把 Top-8 算快了，也把 Top-8 说成了优于 Top-1。
- **去重**：三种口径分开报——题面唯一率 / 答案唯一率 / `(prompt, response)` 字节唯一率。字节去重会让换皮题吃满配额。
- **路由指标**：先按层平均再报；`97.06 / 35 ≈ ln(32)` 是正常值，不是爆炸。
- **旧 README 的乐观数字**：`Code −55% / Math −51% / Arts −40%` 是 in-distribution 结果，不要引用，用 DualBigLittle 的 −9.5%。

## 七、还没定的事（整合前要回答）

- **骨架选谁**：25K（20 簇 × 45 专家）还是 8×16×256 分类学？前者数据最硬，后者结构最整齐。
- **μP 维度自适应缩放**能否从 0.6B 直接迁到 8B+，还是需要重标定。
- **规模-收益拐点**：32 专家（TriTier）→ 896（BigLittle）→ 25,200（Myriad），哪一段开始边际收益塌陷。
- **交付形态**：走 `bake_*` 烘成 stock 权重，还是保留 host 流式层 + 自定义 modeling。
- **后端归属**：25K 的推理服务目前不在仓里，最终要不要收进主线仓。

## 八、维护约定

- 新实验：新开仓 → 这里加一行（变量 / 结果 / 状态），**不改旧行**。
- 结论被推翻：旧行标「已修正」，另起一行写新结论，保留时间顺序。
- 只在本地跑、没推上来的分支不入表，推了再补行。
- 每行都要能指回一个数值来源（README / 脚本 / checkpoint），否则不写数字。

## 九、时间线

| 日期 | 仓库 | 备注 |
| :--- | :--- | :--- |
| 09-28 | BigLittle-MoE | 分层大核 + 896 微专家的第一版 |
| 09-28 | DualBigLittle-MoE | 拆成文理双核 |
| 09-29 | Myriad-MoE | 专家推到 20,000 |
| 09-29 | Myriad-MoE-E2B | 换底座，仅留权重 |
| 09-30 | Myriad-MoE-25K-Micro-Experts | 25,200 专家 + 流式 + 热插拔 |
| 10-01 | Heretic-Scalpel-TriTier-E2B | 三档适配器范式 |
| 10-02 | Heretic-Scalpel-Swarm-Forge | 抽成框架 |
| 10-04 | DeepSeek-Pro-Mini | 从零复现 MLA + MoE |
| 10-05 | myriad-moe-console | 25K 的前端 |
| 10-07 | gemma-4-12B-…-MoE | 大底座上的 16×16 嫁接 |
| 10-10 | myriad-qwen3.5-0.8b | 换分类学（8×16×256） |
| 10-22 | Mini-Architecture | 最小 V4.1 对照实验 |

---

<div align="center">

All numbers above were measured on a single RTX 5090 D unless stated otherwise.
这块台账只服务于一件事：整合的那天，不用重新回忆。

</div>
