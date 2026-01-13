Week 3（章节重点 1）：LLaMA-style Block（现代 LLM 的标准骨架）

面试考点 1：LLaMA 的 Transformer Block 和“经典 Transformer”最大差异是什么？
满分答案：LLaMA 采用 Pre-LN（更准确是 Pre-Norm），并用 RMSNorm 替代 LayerNorm，FFN 用 SwiGLU（门控前馈），位置编码常用 RoPE。这些改动的共同目标是：训练更稳、收敛更好、推理更快、长上下文更可用。

面试考点 2：为什么现代 LLM 更偏向 Pre-LN（Pre-Norm）？
满分答案：Pre-LN 让每个子层输入先归一化，梯度更容易穿透深层网络，训练更稳定，尤其在几十层到上百层时更明显。Post-LN 在深层更容易出现不稳定或收敛困难。

面试考点 3：RMSNorm 是什么？为什么替代 LayerNorm？
满分答案：RMSNorm 只按特征维的 均方根 做缩放（不减均值），计算更简单、速度更快，实践中稳定性也很好。对大模型来说，每一步省一点算子成本就能换来显著吞吐提升。

面试考点 4：LLaMA 为什么经常去掉 Linear 的 bias？
满分答案：bias 对表达能力提升有限，但会增加参数/带来额外算子与访存；在大规模训练与推理中“可有可无”的部分常被移除以提升效率，并简化融合优化（kernel fusion）。

面试考点 5：什么是 weight tying（Embedding 与 LM Head 共享权重）？为什么要做？
满分答案：把输入 embedding 矩阵与输出分类头共享，能减少参数量并提升泛化（相当于对输入输出空间施加一致性约束）。对语言建模来说常见且有效，尤其在中小模型更明显。

Week 3（章节重点 2）：位置编码的现代方案（RoPE 为核心）

面试考点 6：为什么从 Sinusoidal/learned PE 走向 RoPE？
满分答案：RoPE 把位置信息以“旋转”的方式注入到 Q/K 中，本质上更像一种相对位置信息建模，和注意力打分天然耦合。实践上它对长上下文与外推更友好，也更契合 decoder-only LLM 的工程实现。

面试考点 7：RoPE 注入在 Q/K 而不是 token embedding 的意义是什么？
满分答案：把位置直接作用在注意力匹配空间（Q/K）上，使“相对距离”影响打分而不是仅影响表示本身。结果是模型更容易学到“距离相关”的注意模式（例如近邻依赖、长程指代）。

面试考点 8：长上下文外推（context extension）为什么难？核心瓶颈是什么？
满分答案：模型训练时见过的长度分布有限，位置编码（尤其频率分布）与注意力模式会在更长长度上失配，导致注意力退化、重复、跑题等问题。工程上常通过 RoPE 频率缩放/插值等方案缓解，本质是让“位置频率”对更长长度仍可泛化。

面试考点 9：ALiBi 是什么？与 RoPE 的取舍？
满分答案：ALiBi 给注意力 logits 加上与距离相关的线性 bias，结构简单、长外推能力常不错；RoPE 在多数开源 LLM 里更主流，生态更成熟。取舍通常取决于：目标长度、兼容性、实现与既有权重复用。

Week 3（章节重点 3）：Attention 变体（推理吞吐的关键）

面试考点 10：KV Cache 是什么？为什么能显著加速生成？
满分答案：自回归生成每步只新增一个 token，如果每步都重算全部历史的 K/V，会重复做大量计算。KV Cache 缓存历史 K/V，新步只算新 token 的 K/V 并拼接，避免重复计算，因此生成吞吐提升非常明显。

面试考点 11：KV Cache 的主要代价是什么？
满分答案：主要代价是显存/内存占用，随 batch、层数、上下文长度线性增长。长上下文推理常被 KV Cache 的内存瓶颈卡住，所以才需要 MQA/GQA、PagedAttention 等优化。

面试考点 12：MQA（Multi-Query Attention）是什么？解决什么问题？
满分答案：MQA 让多个 Q 头共享更少的 K/V 头（甚至单个），显著减少 KV Cache 的显存占用并提升解码速度。代价是 K/V 表达能力下降，可能带来质量损失，但很多场景是可接受的工程权衡。

面试考点 13：GQA（Grouped-Query Attention）是什么？为什么是 MQA 与 MHA 的折中？
满分答案：GQA 把多个 Q 头分组，每组共享一套 K/V，比 MQA 更保留表示能力，比全 MHA 更省 KV Cache。大模型里常用 GQA 来兼顾质量与吞吐。

面试考点 14：为什么推理阶段常说“prefill”和“decode”是两种不同的性能问题？
满分答案：prefill 是把整段 prompt 一次算完，主要是矩阵乘法吞吐；decode 是逐 token 生成，主要受 KV Cache 访存与 attention 的序列长度增长影响。优化策略也不同：prefill 关注算力利用率，decode 关注内存与 kernel 优化。

面试考点 15：FlashAttention 的核心思想是什么？为什么更快？
满分答案：核心是“分块计算 + 在线 softmax”，避免显式存储完整注意力矩阵，显著降低显存读写（memory IO）。大模型中 attention 往往 IO 受限，FlashAttention 通过减少 IO 提升吞吐。

Week 3（章节重点 4）：FFN 变体（SwiGLU/门控的价值）

面试考点 16：SwiGLU 是什么？为什么比 GELU-MLP 更常见？
满分答案：SwiGLU 是门控前馈：一支做内容、一支做 gate，用 Swish/SiLU 激活后相乘，再投影回去。门控能提高表达效率与梯度流动，实践上在同等算力下往往更强，是 LLaMA 系结构的常见选择。

面试考点 17：为什么 FFN 的隐藏维通常比 d_model 大很多（如 4x/8x）？
满分答案：FFN 是每个 token 的“非线性特征变换”，扩维后再压回能提供更强的表示能力。经验上 FFN 参数占比高，但对性能提升非常关键，是模型容量的重要来源之一。

Week 3（章节重点 5）：Norm / 激活 / 初始化（训练稳定的配方）

面试考点 18：LayerNorm vs RMSNorm：各自的训练稳定性差异怎么答？
满分答案：LayerNorm 做“减均值+除方差”，更标准；RMSNorm 不减均值、只缩放 RMS，计算更省。大规模实践中 RMSNorm 通常能保持稳定同时提升效率，因此在 LLaMA/Qwen 等结构中常见。

面试考点 19：为什么大模型特别重视初始化与数值范围？
满分答案：层数深、序列长、参数量大时，数值范围稍微失控就可能导致梯度爆炸/NaN 或训练不收敛。合理初始化、归一化策略、学习率 warmup、梯度裁剪共同保证训练稳定。

Week 3（章节重点 6）：Tokenizer（SentencePiece/BPE/Unigram）与“对齐到模型”

这块是面试/工作非常容易被忽视但很关键。

面试考点 20：为什么 LLaMA 系常用 SentencePiece？
满分答案：SentencePiece 把分词当作纯文本处理（不依赖空格），对多语言更鲁棒；同时训练/部署流程成熟，易复现。对中文/多语言 LLM 特别重要，因为空格不是可靠边界。

面试考点 21：BPE 与 Unigram 的核心区别？
满分答案：BPE 是从字符开始不断合并高频对；Unigram 是从大词表开始做删减并用概率模型选择子词。工程上两者都常用，关键取决于语料与语言特性；你需要会解释“为什么某模型选这种”。

面试考点 22：为什么 tokenizer 会影响模型能力与成本？
满分答案：tokenization 决定序列长度 T，attention 成本近似 O(T²)，T 变大成本激增；同时子词粒度影响对词形变化、专有名词、中文切分等的表达。好的 tokenizer 能在“短序列 + 可表达”之间取得平衡。

Week 3（章节重点 7）：Qwen / DeepSeek 常见“现代化改动”应怎么讲

这里给你一套不依赖具体版本细节也能拿高分的回答框架（更稳、更通用）。

面试考点 23：你怎么概括 Qwen / DeepSeek 这类现代 LLM 的设计取向？
满分答案：总体取向是“LLaMA-style 主干 + 工程化推理优化 + 训练数据与对齐策略强化”。结构上常见 RMSNorm、RoPE、SwiGLU、GQA/MQA、长上下文扩展；工程上强调吞吐、显存、并行与部署友好。

面试考点 24：MoE（混合专家）是什么？为什么一些模型会用它？
满分答案：MoE 用路由器把不同 token 分配给少量专家网络计算，实现“参数量很大但每 token 计算量可控”。优点是性价比高、扩展到更大参数更容易；难点是负载均衡、训练稳定与推理路由开销。

面试考点 25：如果面试官问“DeepSeek 强在哪里”，你怎么稳妥回答？
满分答案：可以从三点说：训练与数据规模带来的基座能力、结构/并行/推理优化带来的效率、以及对特定任务（如代码/推理/长上下文）的配方强化。避免死背版本参数，强调“设计目标—实现手段—工程收益”。

Week 3（章节重点 8）：从论文到代码（工作能力直接体现）

面试考点 26：读一篇 LLM 架构论文你会抓哪 5 件事？
满分答案：抓（1）Block 结构：Norm/Attn/FFN 的顺序；（2）位置编码类型；（3）Attention 头策略：MHA/GQA/MQA；（4）FFN 类型：GELU/SwiGLU 与维度倍率；（5）推理与训练的工程优化点（KV cache、flash、并行）。这五点足够把大多数模型“还原到代码”。

面试考点 27：你怎么快速定位开源仓库里 attention 的实现？
满分答案：先找 modeling_*.py 或 attention.py，定位 qkv projection、head reshape、mask 处理、softmax、dropout、out projection；再看是否有 KV cache 接口（past_key_values）与是否支持 GQA/MQA。能讲出这些点就是“能改代码的人”。

面试考点 28：面试官追问“你如何验证自己实现的 mask 没 bug”？
满分答案：做三类单测：
1）Causal：确保 token i 的输出不依赖 j>i；
2）Padding：padding 位 attention 权重为 0；
3）广播：batch/head 维变化时行为一致。再用可视化 attention map 或对比参考实现做一致性验证。
