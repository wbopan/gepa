# SPRT: 自适应 Minibatch Gate 方案

## 摘要

本文档探讨在 GEPA 进化式 prompt 优化系统中引入 **SPRT (Sequential Probability Ratio Test)** 来替代固定大小的 minibatch gate。通过 bootstrap 模拟，我们发现当前固定 M=5 的方案效率为负，而自适应采样可以显著提升效率。

---

## 1. 问题背景

### 1.1 GEPA 的 Mutation Loop

GEPA 使用进化算法优化 prompt，每次迭代：

1. **选择父候选** (Parent Selection)
2. **突变** (Mutation): 基于 LLM 反思生成子候选
3. **Minibatch Gate**: 在小批量样本上评估，决定是否进入全量验证
4. **全量验证** (Full Validation): 在完整验证集上评估并决定是否接受

### 1.2 当前方案的问题

当前使用**固定大小的 minibatch** (M ≈ 5) 作为 gate：

```
if sum(child_scores_on_minibatch) > sum(parent_scores_on_minibatch):
    proceed_to_full_validation()  # 花费 K 次评估
else:
    reject_child()
```

**经验数据** (171 个通过 gate 的突变):

| 指标 | Minibatch (M=5) | Validation (K=30) |
|------|-----------------|-------------------|
| 改进率 | 38.7% | **25.7%** |
| 退化率 | 32.5% | **64.9%** |
| 平均变化 | 0.000 | **-0.071** |

**问题诊断**:
- Gate pass rate (42%) 远高于真实改进率 (26%)
- Minibatch 太小，噪声主导，gate 基本是随机的
- 大量差的候选通过 gate，浪费 K 次全量评估

### 1.3 Bootstrap 模拟结果

我们模拟了不同 M 值下的效率：

| M | Gate Pass Rate | Precision | Efficiency |
|---|----------------|-----------|------------|
| 5 (当前) | 41.9% | 39.4% | **-0.0002** |
| 10 | 39.7% | 44.9% | +0.0001 |
| 15 | 38.2% | 48.9% | +0.0002 |
| 20 | 37.2% | 52.0% | +0.0002 |
| **24 (最优)** | 36.6% | 54.1% | **+0.0002** |

**结论**: 固定 M=5 效率为负；最优 M* ≈ 24，但这接近全量验证 (K=30)，说明固定 M 方案本身有局限。

---

## 2. SPRT 算法详解

### 2.1 历史背景

**SPRT (Sequential Probability Ratio Test)** 由 **Abraham Wald** 于 **1943年二战期间**发明，用于军火质量控制。由于其高效性，被美国海军列为机密，1945年战后才解密发表。

**Wald-Wolfowitz 定理 (1948)** 证明了 SPRT 的**最优性**：
> 在给定的 Type I error (α) 和 Type II error (β) 约束下，SPRT 所需的平均样本量是所有假设检验方法中**最小的**。

### 2.2 数学原理

#### 假设检验框架

设我们要比较 child 和 parent，定义：
- **H₀ (零假设)**: Child 并不比 Parent 好，即 child 在非平局样本中的胜率 p = p₀
- **H₁ (备择假设)**: Child 确实比 Parent 好，即胜率 p = p₁ > p₀

#### 对数似然比 (Log-Likelihood Ratio)

对于每个样本 i，观察结果 xᵢ ∈ {+1 (child赢), 0 (平局), -1 (child输)}

累积对数似然比：
$$S_n = \sum_{i=1}^{n} \log \frac{P(x_i | H_1)}{P(x_i | H_0)}$$

具体地：
- Child 赢 (+1): $\Delta S = \log(p_1 / p_0)$
- Child 输 (-1): $\Delta S = \log((1-p_1) / (1-p_0))$
- 平局 (0): $\Delta S = 0$ (不提供区分信息)

#### 停止规则

定义两个阈值：
$$a = \log\left(\frac{\beta}{1-\alpha}\right) \quad \text{(下界，reject)}$$
$$b = \log\left(\frac{1-\beta}{\alpha}\right) \quad \text{(上界，accept)}$$

判定规则：
- 如果 $S_n \geq b$: **Accept H₁** → Child 通过 gate
- 如果 $S_n \leq a$: **Accept H₀** → Child 被拒绝
- 如果 $a < S_n < b$: **Continue** → 继续采样

### 2.3 参数设置

#### 错误率参数 (α, β)

| 参数 | 含义 | 后果 | 推荐值 |
|------|------|------|--------|
| α (Type I) | 误把坏 child 当好 | 浪费 K 次全量评估 | 0.1 - 0.2 |
| β (Type II) | 误把好 child 当坏 | 错失进化机会 | 0.2 |

对于 α = β = 0.2 (80% 置信度):
$$a = \log(0.2 / 0.8) \approx -1.39$$
$$b = \log(0.8 / 0.2) \approx +1.39$$

#### 胜率参数 (p₀, p₁)

| 设置 | p₀ | p₁ | 特点 |
|------|----|----|------|
| 激进 | 0.3 | 0.7 | 只要明显好的，决策快 (1-2样本) |
| **推荐** | 0.4 | 0.6 | 平衡速度和灵敏度 |
| 稳健 | 0.45 | 0.55 | 能捕捉微小改进，但较慢 |

对于 p₀ = 0.4, p₁ = 0.6:
- Child 赢: $\Delta S = \log(0.6/0.4) \approx +0.405$
- Child 输: $\Delta S = \log(0.4/0.6) \approx -0.405$

### 2.4 直观理解

**比喻**：像拔河比赛

| 固定 M Gate | SPRT |
|-------------|------|
| 强制比 5 局，看总分 | 每比完一局就看分数差 |
| 不管战况如何 | Child 连赢 3 局 → 直接判胜 (早停) |
| | Parent 连赢 3 局 → 直接判负 (止损) |
| | 比分焦灼 → 继续加赛 |

**核心优势**：
- 对**明显差**的候选：1-2 个样本就能拒绝，节省评估成本
- 对**明显好**的候选：快速通过，不浪费时间
- 对**势均力敌**的候选：投入更多样本确认

---

## 3. 在 GEPA 中的实现方案

### 3.1 Python 实现

```python
import math
import numpy as np
from dataclasses import dataclass
from typing import Callable, Any
from enum import Enum

class GateDecision(Enum):
    ACCEPT = "accept"      # 通过 gate，进入全量验证
    REJECT = "reject"      # 拒绝 child
    CONTINUE = "continue"  # 继续采样

@dataclass
class SPRTConfig:
    """SPRT Gate 配置"""
    alpha: float = 0.2      # Type I error (误把坏当好)
    beta: float = 0.2       # Type II error (误把好当坏)
    p0: float = 0.4         # H0 假设下的胜率 (差 child)
    p1: float = 0.6         # H1 假设下的胜率 (好 child)
    max_samples: int = 15   # 最大采样数

    def __post_init__(self):
        # 计算停止阈值
        self.threshold_reject = math.log(self.beta / (1 - self.alpha))
        self.threshold_accept = math.log((1 - self.beta) / self.alpha)

        # 计算 LLR 增量
        self.llr_win = math.log(self.p1 / self.p0)
        self.llr_loss = math.log((1 - self.p1) / (1 - self.p0))
        self.llr_tie = 0.0


class SPRTGate:
    """基于 SPRT 的自适应 Minibatch Gate"""

    def __init__(self, config: SPRTConfig = None):
        self.config = config or SPRTConfig()

    def evaluate(
        self,
        eval_func: Callable[[Any, Any], float],
        parent_candidate: Any,
        child_candidate: Any,
        sample_pool: list,
        rng: np.random.Generator = None
    ) -> tuple[bool, int, str]:
        """
        执行 SPRT 序贯测试

        Args:
            eval_func: 评估函数 (candidate, sample) -> score (0 或 1)
            parent_candidate: 父候选
            child_candidate: 子候选
            sample_pool: 可用样本池
            rng: 随机数生成器

        Returns:
            (passed: bool, samples_used: int, reason: str)
        """
        if rng is None:
            rng = np.random.default_rng()

        cfg = self.config
        cumulative_llr = 0.0
        samples_used = 0

        # 随机打乱样本顺序
        indices = rng.permutation(len(sample_pool))

        for idx in indices:
            if samples_used >= cfg.max_samples:
                break

            sample = sample_pool[idx]
            samples_used += 1

            # 评估 parent 和 child
            score_parent = eval_func(parent_candidate, sample)
            score_child = eval_func(child_candidate, sample)

            # 计算差异: +1 (child赢), 0 (平局), -1 (child输)
            diff = int(score_child) - int(score_parent)

            # 更新累积 LLR
            if diff > 0:
                cumulative_llr += cfg.llr_win
            elif diff < 0:
                cumulative_llr += cfg.llr_loss
            # diff == 0: 平局不更新

            # 检查停止条件
            if cumulative_llr >= cfg.threshold_accept:
                return True, samples_used, "Early Accept"

            if cumulative_llr <= cfg.threshold_reject:
                return False, samples_used, "Early Reject"

        # 达到最大样本数，用 LLR > 0 决定 (Greedy 策略)
        decision = cumulative_llr > 0
        return decision, samples_used, "Max Samples (Greedy)"

    def evaluate_batch(
        self,
        parent_scores: list[float],
        child_scores: list[float]
    ) -> tuple[bool, int, str]:
        """
        使用预先计算的分数进行 SPRT 测试
        (适用于已有评估结果的场景)
        """
        cfg = self.config
        cumulative_llr = 0.0

        for i, (s_p, s_c) in enumerate(zip(parent_scores, child_scores)):
            diff = int(s_c) - int(s_p)

            if diff > 0:
                cumulative_llr += cfg.llr_win
            elif diff < 0:
                cumulative_llr += cfg.llr_loss

            if cumulative_llr >= cfg.threshold_accept:
                return True, i + 1, "Early Accept"

            if cumulative_llr <= cfg.threshold_reject:
                return False, i + 1, "Early Reject"

            if i + 1 >= cfg.max_samples:
                break

        return cumulative_llr > 0, i + 1, "Max Samples"
```

### 3.2 集成到 ReflectiveMutationProposer

```python
# 在 reflective_mutation.py 中

class ReflectiveMutationProposer:
    def __init__(self, ..., use_sprt_gate: bool = False, sprt_config: SPRTConfig = None):
        ...
        self.use_sprt_gate = use_sprt_gate
        self.sprt_gate = SPRTGate(sprt_config) if use_sprt_gate else None

    def propose(self, state: GEPAState, ...) -> CandidateProposal:
        ...

        if self.use_sprt_gate:
            # SPRT 自适应采样
            passed, samples_used, reason = self.sprt_gate.evaluate(
                eval_func=lambda candidate, sample: self.adapter.evaluate_single(sample, candidate),
                parent_candidate=curr_prog,
                child_candidate=new_candidate,
                sample_pool=self.trainset.fetch_all(),  # 或使用采样策略
            )

            if not passed:
                # 早停拒绝，节省评估成本
                return None  # 或返回拒绝标记

            # 通过 gate，继续全量验证
            ...
        else:
            # 原有固定 M 方案
            ...
```

### 3.3 配置建议

```python
# 推荐配置
sprt_config = SPRTConfig(
    alpha=0.15,      # 稍严格，避免浪费全量评估
    beta=0.2,        # 允许一定的漏检
    p0=0.4,          # 差 child 的胜率
    p1=0.6,          # 好 child 的胜率
    max_samples=15,  # 最大采样数 (约 K/2)
)
```

---

## 4. 预期收益分析

### 4.1 评估成本节省

假设原方案 M=5，全量验证 K=30:

| 候选类型 | 原方案成本 | SPRT 预期成本 | 节省 |
|----------|-----------|--------------|------|
| 明显差 (60%) | 5 + 30 = 35 | 2 + 0 = 2 | **94%** |
| 势均力敌 (25%) | 5 + 30 = 35 | 10 + 30 = 40 | -14% |
| 明显好 (15%) | 5 + 30 = 35 | 3 + 30 = 33 | 6% |
| **加权平均** | 35 | ~15 | **~57%** |

### 4.2 效率提升

- **减少浪费**: 不再让明显差的候选进入全量验证
- **提高精度**: Gate precision 从 39% 提升到 ~55%
- **自适应**: 根据实际表现动态调整采样量

---

## 5. 相关工作

### 5.1 SPRT 在 AI/ML 中的应用

| 应用 | 论文/工具 | 年份 |
|------|----------|------|
| LLM Self-Consistency | [ConSol](https://arxiv.org/abs/2503.17587) | 2025 |
| A/B Testing | Optimizely, Statsig | 工业界 |
| 临床试验 | 药物测试提前终止 | 经典 |

### 5.2 相关算法

| 算法 | 特点 | 与 SPRT 的关系 |
|------|------|---------------|
| **F-race / irace** | 算法配置的 racing | 类似思想，但用 Friedman test |
| **Successive Halving** | 每轮淘汰一半 | 1 vs N，固定轮次 |
| **Hyperband** | SH + 不同预算分配 | Multi-fidelity |
| **mSPRT** | 参数未知时的 SPRT | 更 robust |

### 5.3 Prompt 进化优化现状

当前主流方法 (EvoPrompt, Promptbreeder, GAAPO) **均使用固定评估**，没有采用自适应采样。

**创新机会**: GEPA + SPRT 可以是一个创新点。

---

## 6. 实验计划

### 6.1 对比实验

| 方案 | 配置 |
|------|------|
| Baseline | 固定 M=5 |
| Fixed-Optimal | 固定 M=24 |
| **SPRT** | α=0.15, β=0.2, p0=0.4, p1=0.6, max=15 |

### 6.2 评估指标

- **Total Budget Efficiency**: 总改进量 / 总评估次数
- **Gate Precision**: 通过 gate 且真正改进的比例
- **Average Samples per Mutation**: 平均每次突变的采样数
- **Final Validation Score**: 最终候选的验证集分数

### 6.3 消融实验

- 不同 (p0, p1) 设置的影响
- 不同 (α, β) 设置的影响
- max_samples 的敏感性

---

## 7. 总结

SPRT 是一个有 80+ 年历史、有数学最优性证明的经典算法。将其引入 GEPA 的 mutation gate 可以：

1. **理论保证**: Wald 定理保证样本效率最优
2. **实践收益**: 预期节省 ~57% 的评估成本
3. **创新价值**: 目前 prompt 进化优化领域尚未有人使用

**下一步**: 实现并集成到 GEPA，进行对比实验验证。

---

## 参考文献

1. Wald, A. (1945). Sequential tests of statistical hypotheses. *Annals of Mathematical Statistics*.
2. Lee, J. et al. (2025). [ConSol: Sequential Probability Ratio Testing to Find Consistent LLM Reasoning Paths Efficiently](https://arxiv.org/abs/2503.17587). *arXiv*.
3. López-Ibáñez, M. et al. (2016). [The irace package: Iterated Racing for Automatic Algorithm Configuration](https://www.sciencedirect.com/science/article/pii/S2214716015300270). *Operations Research Perspectives*.
4. Li, L. et al. (2018). [Hyperband: A Novel Bandit-Based Approach to Hyperparameter Optimization](https://jmlr.org/papers/volume18/16-558/16-558.pdf). *JMLR*.
5. Zöller, M. & Huber, M. (2018). [Termination Detection Strategies in Evolutionary Algorithms: A Survey](https://www.researchgate.net/publication/326826620_Termination_Detection_Strategies_in_Evolutionary_Algorithms_A_Survey). *GECCO*.
