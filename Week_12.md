Week12 面试考点（满分简答版）
1）面试考点：为什么大模型训练一定要分布式？瓶颈到底是什么？

满分简答：
瓶颈主要是 显存（参数+梯度+优化器状态+激活） 和 通信（all-reduce / all-gather / reduce-scatter）。模型越大，显存占用按 O(P) 增长，激活还随序列长度、batch 增长；单卡不够就必须做并行与显存优化。分布式的目标是：把内存与计算拆到多卡，同时把通信开销控制在可接受范围。

2）面试考点：训练时显存到底花在哪？（必背）

满分简答：
四大块：

参数 Parameters（P）

梯度 Gradients（G，和参数同量级）

优化器状态 Optimizer states（O，比如 Adam 额外存 m/v，常是参数的 2 倍）

激活 Activations（A，跟 batch、seq_len、层数强相关）
工程上最常见“炸显存”的不是参数，而是 优化器状态 + 激活。

3）面试考点：Data Parallel（DP）是什么？为什么通信大？

满分简答：
DP：每张卡一份完整模型参数，各自跑不同 mini-batch，然后 梯度 all-reduce 同步。优点实现简单；缺点参数/优化器状态不分摊，显存压力大；并且梯度同步是大头通信。

4）面试考点：ZeRO 是什么？它解决了 DP 的什么问题？

满分简答：
ZeRO（Zero Redundancy Optimizer）把 DP 中“每卡都冗余一份”的状态 分片（shard） 到不同 GPU，从而显著降低显存。核心思想：能分就分，把冗余变成分布式存储。

5）面试考点：ZeRO-1/2/3 区别（100% 会问）

满分简答：

ZeRO-1：分片 优化器状态 O

ZeRO-2：分片 优化器状态 O + 梯度 G

ZeRO-3：分片 优化器状态 O + 梯度 G + 参数 P（最省显存，但 all-gather 参数带来更多通信与复杂度）

**为什么这样分层：**越往后分得越彻底、显存越省，但通信更重、实现更复杂、对吞吐更敏感。

6）面试考点：ZeRO-3 为什么通信更大？它在做什么 all-gather？

满分简答：
ZeRO-3 把参数也分片了，所以每次前向/反向某层需要参数时，需要把该层参数从各 GPU all-gather 到本地临时拼起来，用完再释放/回收。这就是它省显存的代价：更频繁的参数聚合通信。工程上会靠：通信-计算 overlap、bucket、prefetch、分层聚合等缓解。

7）面试考点：ZeRO-Offload / ZeRO-Infinity 是什么？何时用？

满分简答：
Offload 把优化器状态/参数分片放到 CPU 内存或 NVMe，GPU 只保留必要部分。适用于 GPU 显存特别紧 的场景。代价是 PCIe/NVMe 带宽可能成为瓶颈，所以要评估吞吐。结论：能不用 offload 就不用；实在装不下才用。

8）面试考点：混合精度（fp16/bf16）为什么能省显存？有什么坑？

满分简答：
参数/激活用低精度，内存占用直接减半左右，吞吐也提升。坑是数值稳定性：fp16 容易 overflow/underflow，需要 loss scaling；bf16 更稳（动态范围大）但依赖硬件支持。工程建议：能 bf16 就 bf16。

9）面试考点：Activation Checkpointing（激活检查点）是什么？为什么能省显存？

满分简答：
训练反向需要前向激活。Checkpointing 不保存所有激活，只保存少量“检查点”，反向时对中间层 重算前向 换显存。节省显存显著，代价是额外计算。结论：显存不够时是最划算的手段之一。

10）面试考点：Tensor Parallel（TP）是什么？什么时候需要？

满分简答：
TP 把单层矩阵乘拆到多卡（例如将线性层权重按列/行切分），每层 forward 都要通信（all-reduce / all-gather）。当模型单层太大、单卡算不过来或显存放不下单层权重时需要 TP。TP 的通信频繁但可以高效融合到 GEMM 流水。

11）面试考点：Pipeline Parallel（PP）是什么？它的核心 trade-off？

满分简答：
PP 把网络层按深度切成多个 stage，不同 GPU 负责不同层，micro-batch 像流水线一样通过 stage。
trade-off：能训更深更大的模型，但有 pipeline bubble（空泡）；通过增加 micro-batch 数（gradient accumulation）减少空泡。

12）面试考点：3D 并行（DP+TP+PP）怎么组合？怎么选型？

满分简答：

先看 单卡能否放下模型参数/优化器/激活：不行 → ZeRO/Checkpoint/Offload

再看 单卡算力是否够：不够 → TP/PP

最后用 DP 扩大吞吐
典型组合：TP×PP 解决模型规模，DP（或 ZeRO）解决吞吐与剩余显存。工程上优先从 ZeRO + checkpointing + bf16 开始，TP/PP 作为第二阶段复杂化。

13）面试考点：为什么梯度累积（grad accumulation）在分布式里更重要？

满分简答：
它能在不增加显存的前提下增大“有效 batch”，同时在 PP 里减少 bubble，稳定训练；代价是更新频率下降、训练步更慢。

14）面试考点：分布式训练最常见的坑与排障思路（非常加分）

满分简答：

NCCL hang：多进程不同步/网络问题/死锁 → 先开 NCCL_DEBUG=INFO，检查 rank 进度、确保每步 all-reduce 都对齐。

显存忽高忽低：激活/缓存/不当 bucket → 开 checkpointing、调小 bucket_size、看是否 OOM 在 backward。

吞吐很差：offload/通信占用/IO → 关 offload 试、增大 batch/micro-batch、开启 overlap、提升数据加载。

loss NaN：fp16 不稳 → 换 bf16、调小 lr、加 warmup、开启 gradient clipping。
