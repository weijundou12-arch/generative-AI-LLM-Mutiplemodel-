Week10 知识点（面试满分答案模式）
1）面试考点：RLHF 是什么？解决什么问题？为什么 SFT 不够？

满分回答：
RLHF（Reinforcement Learning from Human Feedback）是用“人类偏好”作为奖励信号，把模型从“会回答”对齐到“更符合人类期望地回答”（更有用、更安全、更符合格式/语气）。SFT 只能模仿数据分布：如果数据里混有噪声/风格不一致/对安全与真实缺乏约束，模型会学到“看起来像对的答案”，但不一定满足用户偏好与安全目标；RLHF 通过奖励优化，把“更好”的答案概率显式拉高。

2）面试考点：RLHF 的标准三段式流程是什么？每段为什么存在？

满分回答：
典型三段：SFT → Reward Model（RM）→ RL（常用 PPO）。

SFT：先让模型具备基本指令跟随与可用性，否则 RL 会在很差的分布上探索，成本高且不稳定。

RM：人类打分贵，把“人类偏好比较（A 优于 B）”拟合成一个可自动打分的奖励模型。

RL(PPO)：用奖励来更新策略，同时用 KL/clip 约束更新幅度，避免模型为了奖励过度漂移（灾难性遗忘/奖励黑客）。

3）面试考点：Reward Model 怎么训练？用的是什么损失？为什么是 pairwise？

满分回答：
RM 通常用**成对偏好数据（chosen / rejected）**训练：给同一 prompt 的两个回答打分，让 chosen 的分数 > rejected。常见是 Bradley–Terry / logistic pairwise loss：

LRM​=−logσ(rθ​(y+)−rθ​(y−))

用 pairwise 是因为人类更擅长“比较哪个好”，而不是给绝对分；且比较标注一致性更高。

4）面试考点：为什么 PPO 是 RLHF 里最常见的 RL 算法？关键稳定性机制是什么？

满分回答：
PPO 常用是因为实现成熟、对超参相对稳健，并且通过 clipping 限制策略更新幅度，训练不易崩。RLHF 里还会加 KL penalty（让新策略别偏离参考模型/ SFT 模型太远），用来抑制“为了高奖励乱说话”的漂移。稳定性的核心就是：“既追求奖励提升，又约束分布漂移”。

5）面试考点：RLHF-PPO 里的 reward 是怎么构造的？为什么要加 KL？

满分回答：
实际优化的常是“奖励模型分数 − KL 罚项（相对参考策略）”。

RM 分数推动“更符合偏好”；

KL 罚项防止模型走到参考分布之外（减少胡编、减少遗忘、减少 reward hacking）。
KL 也能理解为：我们在做“带先验约束的偏好优化”，先验就是 SFT 模型。

6）面试考点：PPO 里 actor / critic 分别干嘛？优势函数（advantage）为什么重要？

满分回答：

Actor（policy）：生成回答，更新目标是让高奖励回答概率更高。

Critic（value）：估计期望回报作为 baseline，减少方差，让训练更稳。
优势函数 
A=R−V 表示“比预期好多少”，用它更新可以显著降低梯度方差，否则训练非常抖、样本效率差。

7）面试考点：DPO 是什么？它为什么能“不要 RM、不要 PPO”？

满分回答：
DPO（Direct Preference Optimization）直接用偏好对（chosen/rejected）优化策略，目标等价于带 KL 正则的 RLHF 目标的一种闭式形式：不用显式训练 RM，也不用在线 rollouts 的 PPO。它的核心损失就是让策略对 chosen 的相对对数概率（相对 reference policy）更大：

LDPO​=−logσ(β[(logπ(y+)−logπ(y−))−(logπref​(y+)−logπref​(y−))])

优点是工程简单、稳定、成本低；缺点是偏好信号更“静态”，不如 RL 那样灵活接入复杂奖励。

8）面试考点：DPO 里的 β 是什么？调大/调小会怎样？

满分回答：
β 控制“偏好推动强度/等价 KL 约束强度”。

β 大：更新更激进，更快贴近偏好，但更易过拟合偏好数据、风格变形。

β 小：更保守，效果更稳但提升慢。

9）面试考点：GRPO 是什么？和 PPO 最大区别是什么？为什么它在推理类任务火？

满分回答：
GRPO（Group Relative Policy Optimization）是 PPO 的变体：去掉 critic（value model），改成“同一 prompt 采样一组回答”，用组内平均分/相对排序做 baseline：

对每个 prompt 采样 K 个回答 → 逐个打分 → baseline 用组均值 → advantage = r_i − mean(r)。
这样省掉 critic 训练，资源更省；在推理任务（数学/代码/可验证问题）中，往往能通过规则/验证器打分，训练效率更高。DeepSeekMath 和 DeepSeek-R1 都使用了 GRPO 思路来提升推理能力。

10）面试考点：什么是 RLVR？它和 RLHF 的关系？

满分回答：
RLVR（Reinforcement Learning from Verifiable Rewards）是“奖励来自可验证规则/工具”，比如数学答案校验、编译器/单元测试、JSON schema 校验。它可以和 GRPO 搭配：不需要人工偏好，也不一定需要 RM，而是用验证器直接给 reward。推理模型训练里很常见。

11）面试考点：RLHF 训练里最常见的失败模式有哪些？怎么诊断/修？

满分回答：

Reward hacking：模型学会投机取巧骗 RM（堆砌关键词/套模板）。→ 加 KL、混合 SFT 数据（PPO-ptx 思路）、改 RM 数据覆盖、引入反例。

Length bias：RM 偏爱长答案。→ reward 做长度归一/惩罚，或在数据里控制长度。

Mode collapse：输出同质化。→ 增强探索（温度/采样）、多样性约束、调 β / KL。

分布漂移：过度对齐导致常识能力下降。→ 更强 KL、定期评测基准集、混合预训练数据。

12）面试考点：工业落地时，PPO vs DPO vs GRPO 怎么选？

满分回答：

DPO：首选（数据是偏好对、追求稳定低成本、迭代快）。

PPO(RLHF)：当奖励结构复杂/需要在线交互/需要更强的“策略优化”能力。

GRPO/RLVR：当任务有可验证 reward（数学、代码、格式严格、工具链验证）且想省掉 critic；推理/工具型对齐很合适
