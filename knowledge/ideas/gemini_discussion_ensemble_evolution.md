# 与 Gemini 讨论：Ensemble-aware GEPA 设计

> 讨论日期: 2026-02-01
> Session ID: 366085b2-4c02-47c0-a644-816e3f75eb64
> 目标：将 GEPA 从"单体优化"改进为"集成优化"，产出一组互补的 candidates

---

## 背景

当前 GEPA 框架输出单一最优候选者。目标是改进框架，使其能够：
- 产出一组互补的 candidates
- 每个 candidate 擅长不同类型的样本
- 集成结果（Oracle ensemble）最优

---

## 第一轮：核心架构设计

### 1. 集成架构（Runtime）

**推荐方案：Router-based MoE + Greedy Set Cover**

- **Learned Router**：训练轻量分类器根据 Query 预测最佳 Expert
- **Embedding-based k-NN**：无需训练，利用预训练 Embedding 的语义聚类能力
- **Confidence Cascade**：先用通用 Prompt，置信度低时再调用专家

### 2. 优化目标修改

**从个体最优 → 边际贡献（Marginal Contribution）**

$$Fitness(C | Archive) = OracleScore(Archive \cup \{C\}) - OracleScore(Archive)$$

- 如果一个 Candidate 只能答对大家都能答对的题，边际贡献 = 0
- 如果它能答对所有人都答错的题，价值极高

### 3. 反射变异目标修改

- 只关注"现有集成无法解决的错题"
- Prompt 调整：鼓励产生极度偏科的"特种兵"Prompt

### 4. Pareto 前沿的利用

**Greedy Set Cover 算法**：
1. 选均分最高的 C1（Base Model）
2. 移除 C1 答对的列
3. 选覆盖最多残差的 C2
4. 重复直到选出 k 个或残差为 0

---

## 第二轮：关键细节

### 1. 边际贡献的计算时机

**问题**：Archive 不断变化，早期候选者可能变得冗余

**解决方案：分离存档与选择**
- **存档（Archiving）**：基于 Pareto 支配（静态）- 宽容
- **选择（Selection）**：基于边际贡献（动态）- 功利

### 2. Router 冷启动问题

**解决方案：Embedding-based k-NN（非参数方法）**
1. 对验证集 Query 计算 Embedding
2. 记录每个 Query 被哪个 Expert 解决最好
3. 推理时找 k 个最近邻，加权投票

### 3. 零正确率题目处理

**延迟激活 + 资格测试**：
- 前 50% 阶段：忽略 0% 正确率题目
- 后 50% 阶段：解锁，但要求通过"基线回归测试"

$$Cost = Score(BaseModel) - Score(C \text{ on EasySet})$$

### 4. 连续分数处理

**Greedy Max-Sum 算法**：

$$Gain(c) = \sum_{x \in D} \max(0, Score(c, x) - CurrentMaxScore[x])$$

自动处理连续值，考虑边际分数提升而非仅覆盖数量。

---

## 第三轮：代码实现

### 1. 修改后的进化循环

```python
while not should_stop():
    # 1. 父代选择 - 选择最有希望突破盲区的
    parent_idx = ensemble_selector.select_parent(state, current_max_scores)

    # 2. 采样 - 优先盲区样本
    minibatch_ids = blind_spot_sampler.sample(state, current_max_scores)

    # 3. 变异 - 传入集成上下文
    proposal = ensemble_proposer.propose(parent, eval_result, current_max_scores)

    # 4. 接受 - 保持 Pareto 非受支配
    if self._is_pareto_improvement(proposal, state):
        state.add_candidate(...)
        update_current_max_scores(state.pareto_front)
```

### 2. Anchor + Target 反射数据集

```python
def make_reflective_dataset_for_ensemble(
    candidate, eval_batch, current_ensemble_coverage, target_ratio=0.7
):
    # Target: 当前集成做错的 && 本次评估也做错的
    targets = [s for s in eval_batch if current_ensemble_coverage[s.id] < threshold]

    # Anchor: 当前父代能做对的（基本盘）
    anchors = [s for s in eval_batch if s.score > high_threshold]

    # 混合采样
    return build_dataset(
        sample(targets, n_targets) + sample(anchors, n_anchors)
    )
```

### 3. 推荐指标

| 指标 | 意义 |
|-----|------|
| Oracle Score | 种群潜力的理论上限 |
| Top-k Oracle Score | 实际部署效果 |
| Unique Solve Rate | 互补性强弱 |

---

## 第四轮：工程细节

### 1. Coverage 更新频率

**推荐：每次接受新候选者时立即更新**

- LLM 调用很慢，震荡不是主要问题
- 信息时效性更重要
- 使用增量更新，计算成本极低

### 2. Greedy Set Cover 时机

- 每 10 步运行一次，仅用于监控 Oracle Score 趋势
- 不要用来干预 Pareto 存档的生存权

### 3. 特种兵变异 Prompt

```markdown
You are optimizing a prompt to act as a **Specialist**.

**DATA:**
[SET A: TARGET FAILURES] - The Generalist FAILS here. You MUST solve these.
[SET B: ANCHOR SUCCESSES] - The Generalist SUCCEEDS here. Try to maintain.

**INSTRUCTIONS:**
1. **ANALYZE**: What makes SET A hard? Domain? Format? Trick type?
2. **STRATEGIZE**: Propose modification targeting this characteristic.
3. **GENERATE**: Output the full improved prompt.
```

### 4. Router 策略

**推荐：Confidence Cascade + Embedding Router**

1. 先跑通用专家 A
2. 快速检查置信度
3. 如果低置信度 → Embedding 路由到专家

### 5. 数据泄露防护

**三级划分**：
- Test Set (20%): 绝对不碰
- Evolution Set (80%): GEPA 进化
- 集成选择在 Evolution Set 上，最终评估在 Test Set 上

---

## 第五轮：实施路线图

### 实施优先级

| Phase | 内容 | 目的 |
|-------|-----|------|
| **1 (MVP)** | Oracle Score 追踪 + Greedy Max-Sum | 验证隐式互补性是否存在 |
| **1.5** | 手动验证集成效果 | 确认互补性是真实的 |
| **2 (Core)** | BlindSpotSampler + Coverage 追踪 | 让进化关注盲区 |
| **3 (Advanced)** | SpecialistProposer + EnsembleSelector | 显式引导专业化 |
| **4 (Deploy)** | Embedding Router + Cascade | 部署架构 |

### 验证指标

| Phase | 指标 | 判据 |
|-------|-----|------|
| 1 | Ensemble Gain | `(Oracle - Best) / Best > 5%` |
| 2 | Coverage Entropy | 盲区覆盖速度应加快 |
| 3 | Anchor Retention | 特种兵变异后基线保持率 |
| 4 | Net Performance | 考虑 Router 损耗后 > 单体最优 |

### 潜在失败模式

1. **模型能力边界**：某些题超出 LLM 能力
   - 对策：Hard Negative 黑名单机制

2. **虚假互补**：随机性导致的假阳性
   - 对策：多次采样确认稳定性

3. **Router 开销抵消收益**
   - 对策：坚持 Confidence Cascade

### 定位

| 维度 | GEPA-Ensemble | MoE | Algorithm Ensemble |
|-----|---------------|-----|-------------------|
| 层级 | Prompt Space | Weight Space | Decision Space |
| 数据需求 | 极低 (100-200) | 极高 (Billion) | 中等 |
| 可解释性 | 高 | 低 | 低 |
| 部署灵活性 | 极高 | 低 | 中等 |

**核心优势**：GEPA 是"黑盒模型"的"白盒补丁"，在无法修改模型权重时，是提升性能上限的唯一自动化手段。

---

## 核心代码改动清单

### 新增文件

1. `utils/diversity.py`
   - `compute_performance_vector()`
   - `compute_oracle_score()`
   - `greedy_max_sum_selection()`

2. `strategies/blind_spot_sampler.py`
   - `BlindSpotSampler` 类

3. `strategies/ensemble_selector.py`
   - `EnsembleAwareSelector` 类

4. `proposer/specialist_proposer.py`
   - `SpecialistMutationProposer` 类

### 修改文件

1. `core/state.py`
   - 新增 `current_max_scores` 字段
   - 新增 `update_coverage()` 方法

2. `core/engine.py`
   - 结束时调用 `greedy_max_sum_selection()`
   - 返回 `ensemble` 字段

3. `core/result.py`
   - 新增 `ensemble` 和 `router_data` 字段

---

## 关键结论

1. **Phase 1 是关键决策点**：如果 Oracle Score 仅比 Best Single 高 1%，说明任务本身不可分，后续投入可能不值得

2. **BlindSpotSampler 是灵魂**：将进化算力从"内卷"转移到"开拓"

3. **保持简单**：如果 Phase 1 发现 Oracle Score 很高，直接跳到 Phase 4 做 Router 可能就够用

4. **数据隔离**：必须锁死 20% Test Set，集成方法最容易过拟合验证集
