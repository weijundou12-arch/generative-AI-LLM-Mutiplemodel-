Week 1：Python 与深度学习底层（训练能跑 + 能定位 bug + 能解释稳定性）
章节重点 1：张量/Shape 推理

面试考点 1：Transformer 常见张量形状（B,T,D）是什么？
满分答案：B 是 batch size，T 是序列长度（token 数），D 是隐藏维度（embedding/hidden size）。Transformer 的绝大多数 bug 来自维度对不上或广播错误，所以必须能对每层输入输出形状做推导。

面试考点 2：矩阵乘法/广播规则为什么重要？
满分答案：Attention 的核心计算是 Q @ K^T，形状从 (B,H,T,Dh) @ (B,H,Dh,T) 变成 (B,H,T,T)。mask 往往要广播到 (B,H,T,T)，广播错了会出现“看未来/看 padding”的隐性错误。

面试考点 3：reshape/view/transpose/permute 区别是什么？为什么 contiguous 重要？
满分答案：transpose/permute 只是改变 stride，可能导致张量非连续；view 需要连续内存，否则会报错或产生错误视图；contiguous 会拷贝生成连续内存，保证后续 view/线性层安全。多头注意力经常在 split/merge heads 时踩这个坑。

章节重点 2：数值稳定性（NaN 排查必备）

面试考点 4：softmax 为什么会数值不稳定？如何解决？
满分答案：softmax 有 exp(x)，x 大时 exp 溢出变 inf，导致 NaN。标准解法是减去最大值：softmax(x)=exp(x-max)/sum(exp(x-max))，数值等价但稳定。

面试考点 5：为什么 attention mask 用 -inf（或很小负数）？
满分答案：把禁止位置的 logits 设为 -inf，softmax 后概率严格为 0，且不会影响其它位置归一化；这是实现 causal mask / padding mask 的标准方式。

面试考点 6：logsumexp trick 是什么？为什么等价？
满分答案：log(sum(exp(x))) 不稳定，写成 m + log(sum(exp(x-m)))，其中 m=max(x)，利用指数平移不改变归一化比例，避免 overflow。

章节重点 3：反向传播直觉与训练稳定性

面试考点 7：为什么深网络会梯度消失/爆炸？残差如何解决？
满分答案：多层复合导致梯度是多项导数连乘，容易趋近 0 或爆炸。残差连接提供“恒等捷径”，让梯度可以绕开复杂变换直接传播，显著提升深层可训练性。

面试考点 8：dropout 的作用是什么？train/eval 为什么要切换？
满分答案：dropout 随机置零部分激活，减少共适应、防过拟合。训练时启用，推理时要关闭保证确定性与性能，所以需要 model.train() / model.eval()。

章节重点 4：优化器与学习率策略（工程高频）

面试考点 9：Adam vs SGD，为什么大模型常用 AdamW？
满分答案：Adam 用一阶/二阶动量自适应学习率，收敛更快更稳；AdamW 把 weight decay 从梯度更新里解耦，避免 Adam 下 L2 正则效果偏差，是 Transformer/LLM 的标准优化器。

面试考点 10：为什么 Transformer 训练需要 warmup？
满分答案：早期参数随机、梯度噪声大，直接用高学习率易发散。warmup 让学习率从小到大逐步升高，提高训练稳定性。

面试考点 11：什么是 gradient clipping？为什么有用？
满分答案：对梯度范数设上限（如 1.0），防止偶发的大梯度导致参数爆炸与 loss NaN，常用于 RNN/Transformer 的稳定训练。

章节重点 5：PyTorch 基础（面试+工作底线）

面试考点 12：Parameter vs buffer 区别？
满分答案：Parameter 会被优化器更新（可训练参数）；buffer 不参与训练但会随模型保存/加载与搬迁 device（如位置编码表、running stats）。

面试考点 13：no_grad 的作用是什么？
满分答案：推理时关闭 autograd，不保存中间激活，显著节省显存并加速，适用于评估与生成。

Week 2：Transformer 核心（你要能解释并手写关键模块）
章节重点 1：为什么需要 Attention（结构动机）

面试考点 14：为什么 Transformer 取代 RNN/CNN？
满分答案：RNN 串行计算难并行、长依赖差；CNN 需要很多层才能覆盖长距离。Attention 能并行计算任意位置交互，且更适配 GPU，长程依赖建模更直接。

章节重点 2：Scaled Dot-Product Attention（核心公式）

面试考点 15：Attention 的公式与每项含义？
满分答案：Attn(Q,K,V)=softmax(QK^T/√d + mask) V。Q 表示当前查询，K 表示可匹配的“索引”，V 是要汇聚的信息内容；softmax 得到权重后对 V 加权求和。

面试考点 16：为什么要除以 √d？
满分答案：维度增大时 dot product 方差增大，softmax 输出变得极尖，梯度不稳定；除以 √d 进行尺度归一化，稳定训练。

面试考点 17：causal mask 与 padding mask 有何区别？
满分答案：causal mask 禁止看未来 token（自回归生成必须）；padding mask 禁止关注 padding 位置（变长序列的空位）。二者可叠加使用。

章节重点 3：Q/K/V 为什么要分开学（高频灵魂题）

面试考点 18：为什么不直接用 embedding 做相似度，而要投影成 Q/K/V？
满分答案：投影让模型在不同子空间学习不同关系：Q 决定“我想要什么信息”，K 决定“我能提供什么线索”，V 决定“我提供的信息内容”。分离后表达能力更强，也使多头注意力可在不同投影空间并行建模多种关系。

章节重点 4：Multi-Head Attention（为什么多头有效）

面试考点 19：多头注意力的动机是什么？
满分答案：单头相当于在一个表示空间里做匹配，容易受限；多头是多个低维子空间并行关注不同模式（局部/全局、语法/语义、指代等），提升表达能力与泛化。

面试考点 20：多头计算流程？
满分答案：先用线性层得到 Q/K/V（维度 D），再 reshape 为 H 个 head（Dh=D/H），每个 head 独立 attention，最后 concat 并线性投影回 D。

章节重点 5：位置编码（为什么必须要）

面试考点 21：为什么 Transformer 需要位置编码？
满分答案：注意力对输入 token 的排列天然不敏感（置换不变），不加位置信息模型无法区分顺序。位置编码把“顺序”注入 token 表示，才能建模语序与依赖。

面试考点 22：sinusoidal vs learned 位置编码区别？
满分答案：sinusoidal 是固定函数编码，具备一定外推能力；learned 是可训练参数，更贴合数据分布但外推不一定好。现代 LLM 多用 RoPE/ALiBi 等更适合长上下文的方法（Week3 深入）。

章节重点 6：FFN（非线性变换的必要性）

面试考点 23：FFN 在 Transformer 里干什么？
满分答案：Attention 更像“信息路由/加权汇聚”，FFN 对每个 token 独立做非线性特征变换，提升模型表示能力。没有 FFN，模型表达会明显受限。

章节重点 7：LayerNorm + Residual（训练可行性的关键）

面试考点 24：为什么用 LayerNorm 而不是 BatchNorm？
满分答案：序列任务 batch size 可能很小、长度变化大，BatchNorm 依赖 batch 统计量易不稳定；LayerNorm 对单样本特征维归一，适配 NLP/Transformer。

面试考点 25：残差连接为什么关键？
满分答案：让梯度和信息通过恒等路径传播，减轻深层训练困难；同时让每层学习“增量修正”而非从零重构。

章节重点 8：Pre-LN vs Post-LN（深层训练高频）

面试考点 26：Pre-LN 和 Post-LN 区别？为什么现代 LLM 多用 Pre-LN？
满分答案：Post-LN 是原始 Transformer（LN 在残差之后），深层训练更不稳定；Pre-LN（LN 在子层之前）梯度更稳定，更易训练深网络，因此现代大模型普遍采用 Pre-LN。

章节重点 9：Decoder-only GPT（为什么主流 LLM 是这样）

面试考点 27：为什么 GPT/LLaMA/Qwen 多采用 decoder-only？
满分答案：decoder-only 用 causal mask 做自回归 next-token 预测，目标统一、数据规模易扩展、适配生成任务。相比 encoder-decoder，它更适合“通用生成式建模”。

章节重点 10：训练目标与评估指标

面试考点 28：next-token prediction 如何构造 label？
满分答案：输入序列 x 的第 0..T-1 个 token，预测第 1..T 个 token（label 右移一位）。loss 通常用 cross-entropy 直接从 logits 计算。

面试考点 29：perplexity 是什么？
满分答案：ppl = exp(cross_entropy_loss)，衡量模型对序列的不确定性；越低说明模型越能预测下一 token。

章节重点 11：推理与加速入门（KV Cache）

面试考点 30：KV Cache 是什么？为什么能加速生成？
满分答案：生成时每步只新增一个 token，如果每次都对全部历史重算 K/V，代价高。KV cache 把历史 K/V 缓存起来，新步只计算新 token 的 K/V，与历史拼接即可，避免重复计算，显著提升吞吐。
