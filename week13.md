Week13 面试考点（满分简答版）
1）面试考点：推理为什么慢？主要瓶颈是什么？

满分简答：
推理慢通常不是“模型算不动”，而是 内存带宽与缓存 限制：

Prefill（提示阶段）：主要是大矩阵乘（GEMM）+ attention 计算，吞吐相对高；

Decode（逐 token 生成）：每步只生成 1 个 token，但要做全模型前向，且 attention 需要读取越来越大的历史 KV，变成 memory-bound，吞吐下降明显。
所以推理优化核心是：减少每 token 的计算与内存读写，尤其是 KV cache 的管理与访问。

2）面试考点：KV Cache 是什么？为什么能加速？

满分简答：
自回归生成时，每一步 attention 都要用到历史的 Key/Value。KV cache 把历史 K/V 缓存起来，避免每步重复计算历史 token 的 K/V，从而把每步复杂度从“重新算全序列”降为“只算新 token 的 K/V + 与历史 cache 做 attention”。
**为什么重要：**decode 阶段的主成本就来自“不断增长的历史”，KV cache 是让长文本推理可用的基础。

3）面试考点：KV cache 的显存怎么估算？（必背）

满分简答：
近似：

KV bytes≈2×L×S×H×D×bytes_per_elem

2：K 和 V

L：层数

S：序列长度（context + 已生成）

H：头数

D：每头维度

dtype：fp16/bf16 2 bytes，fp32 4 bytes
**结论：**长上下文+高并发时，KV cache 常是显存第一杀手。

4）面试考点：为什么 decode 阶段吞吐会随长度下降？

满分简答：
因为 attention 要对历史 token 做加权，历史越长，读取 KV 越多、softmax 越大，且每步生成都要重复；即使算子是 flash-attn，decode 仍会受到 cache 读写与 kernel launch 的限制，属于 memory-bound。

5）面试考点：PagedAttention 是什么？解决什么问题？

满分简答：
PagedAttention（vLLM 的代表做法）把 KV cache 像操作系统的“分页内存”一样管理：KV 不要求连续大块显存，而是分成固定大小 block，并用页表映射每个序列的 block。
**解决的问题：**传统连续 KV cache 会产生严重碎片和浪费，尤其是不同请求长度不一、并发动态变化时。PagedAttention 通过分页复用显存，大幅提高 KV 利用率与吞吐。
**一句话：**它是为“高并发长上下文推理”而生的 KV 内存管理。

6）面试考点：Continuous Batching（连续批处理）是什么？为什么比静态 batch 好？

满分简答：
静态 batch 要等一批请求凑齐再算，且长度不一会 padding 浪费；Continuous batching 允许在 decode 过程中动态加入/退出请求，把 GPU 一直喂满，减少空转。
**为什么重要：**生产推理请求是流式到达的，连续批处理显著提升吞吐与 p95 延迟表现，是推理服务的核心能力之一（vLLM 就靠这个吃饭）。

7）面试考点：推理量化有哪些？INT8 / INT4 的取舍？

满分简答：
量化用低比特表示权重（有时也量化激活），减少显存与带宽，提高吞吐。

INT8：精度损失小，上线稳定，常用于通用模型

INT4（GPTQ/AWQ）：更省显存，能上更大模型，但精度更敏感、算子依赖更强
取舍：优先满足业务指标（准确性/安全），在可接受损失下追求更高吞吐与更低成本。

8）面试考点：AWQ vs GPTQ 的直觉区别？

满分简答：
两者都是 4bit 权重量化思路。直觉上：

GPTQ：更偏“二阶/重构误差最小化”的离线量化，常见于离线校准

AWQ：强调“激活重要性/权重保护”，常在保持精度上更稳（很多工程实践偏好）
面试答法：不需要背细节公式，但要讲清“都在用校准数据降低量化误差，差异在误差建模与重要性度量”。

9）面试考点：FlashAttention 在推理里解决什么？

满分简答：
FlashAttention 通过 IO-aware 的 block 计算减少 HBM 读写和中间张量存储，使 attention 更接近算力上限。prefill 阶段收益更明显；decode 阶段仍会受 KV cache 读取与 kernel 调度影响，但依然有收益。

10）面试考点：Speculative Decoding 是什么？何时有效？

满分简答：
用一个小 draft 模型一次生成多个 token，大模型验证并接受其中一段，减少大模型 forward 次数。
有效条件：draft 模型要足够快、且与大模型输出一致率高；否则验证成本抵消收益。适合高吞吐场景、且模型家族一致（同域）时。

11）面试考点：蒸馏（distillation）在推理加速里的角色？

满分简答：
蒸馏把大模型能力迁移到小模型（或 MoE 路由简化），直接减少推理成本。
量化是“同模型更快”，蒸馏是“更小模型替代”。工程上常组合：先蒸馏得到更小模型，再做 INT8/INT4。

12）面试考点：推理系统的关键指标有哪些？你会怎么权衡？

满分简答：

吞吐：tokens/s、requests/s

延迟：p50/p95/p99（prefill 与 decode 分开看）

成本：每 1k tokens 成本、GPU 利用率

质量：任务指标、幻觉率、安全拒答率
权衡：不同业务偏好不同（客服看延迟，批量总结看吞吐）；优化必须在“质量不崩”的前提下做。
