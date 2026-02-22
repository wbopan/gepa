# NAS-Inspired Memory Architecture Search for GEPA

> 分析日期: 2026-02-22
> 基于: Memory Adapter 代码分析 + NAS/LLM 记忆系统文献调研

---

## 1. 问题定义

### 1.1 当前 Memory Adapter 的效率瓶颈

GEPA 的核心进化循环依赖 **minibatch gate** 实现高效的候选筛选：

```
Prompt 优化 (高效):
  父代 minibatch 评估:     3 个样本 (with traces)
  子代 minibatch 评估:     3 个样本 (no traces)
  通过 gate → 全量验证:   ~500 个样本
  拒绝率 ~80% → 平均每轮: ~106 个样本
```

但 **Memory Adapter** 面临根本性的效率困境：

| 维度 | Prompt 优化 | Memory 优化 |
|------|-----------|------------|
| **生成** | 1-2 个样本的反馈即可生成新 prompt | 需要整个数据集的反馈才能生成合理的记忆 |
| **评估** | 3 个样本的 minibatch 就能判断好坏 | 记忆的价值只有在整个数据集上才能体现 |
| **变异粒度** | 整体重写或 patch | 单条 entry 的 edit (CREATE/UPDATE/DELETE) |
| **信号稀疏性** | 每个样本都能给出有意义的信号 | 单个样本对记忆系统的评价噪声极大 |

**根本原因**: Prompt 是一个**全局生效**的指令系统 — 改一个词就影响所有样本。而 Memory 是一个**局部生效**的知识系统 — 一条记忆可能只对特定类型的问题有用。因此：

1. **生成需要全局视角**: 要知道哪些知识缺失，必须看到足够多的失败案例
2. **评估需要全局覆盖**: 一条新记忆可能只改善 5% 的样本，在 3 个样本的 minibatch 中很可能完全不被观察到
3. **Minibatch gate 几乎无效**: 随机 3 个样本中包含受影响样本的概率很低，gate 变成随机抛硬币

### 1.2 形式化问题

设记忆系统 $M = \{e_1, e_2, ..., e_k\}$ 由 $k$ 个 entry 组成。数据集 $D = \{d_1, ..., d_n\}$。

对于每个样本 $d_i$，只有部分 entry 是相关的：$\text{relevant}(d_i) \subset M$，其中 $|\text{relevant}(d_i)| \ll |M|$。

**关键洞察**: 修改 entry $e_j$ 只影响 $\text{affected}(e_j) = \{d_i : e_j \in \text{relevant}(d_i)\}$。

如果 $|\text{affected}(e_j)| / |D| = 5\%$，那么在 minibatch_size=3 的情况下：
$$P(\text{至少 1 个受影响}) = 1 - (0.95)^3 = 0.143$$

即 **85.7% 的概率** minibatch 中没有任何受影响的样本，gate 完全无法检测到变化。

---

## 2. 核心思路: 将记忆系统视为一个可搜索的架构

### 2.1 类比: Memory System ≈ Neural Architecture

| NAS 概念 | Memory System 对应 |
|---------|-------------------|
| 神经网络架构 | 记忆系统的抽取/检索管道 |
| 操作 (conv, pool, etc.) | 抽取操作 (summarize, extract_entities, classify, etc.) |
| 层/Block | 抽取的层级 (raw → structured → abstracted) |
| 超网络 (Supernet) | 包含所有可能抽取路径的记忆管道 |
| 子网络采样 | 选择特定的抽取路径 |
| 权重共享 | 共享底层抽取结果 |
| Zero-cost proxy | 不需要完整评估的记忆质量指标 |

### 2.2 记忆系统作为层级抽取管道

将记忆系统从"扁平的 key-value store"重新定义为一个**层级抽取架构**：

```
Layer 0 (Raw Data):
  └─ 原始数据集样本 {d_1, ..., d_n}

Layer 1 (Instance-level extraction):
  └─ 每个样本的关键信息提取
  └─ 操作选择: [verbatim_copy, key_fact_extraction, pattern_extraction, error_analysis]

Layer 2 (Cluster-level aggregation):
  └─ 将相似样本的信息聚合
  └─ 操作选择: [topic_clustering, difficulty_grouping, error_type_clustering]

Layer 3 (Global abstraction):
  └─ 生成全局性的知识/规则
  └─ 操作选择: [rule_induction, common_pattern_synthesis, exception_cataloging]

Layer 4 (Memory compilation):
  └─ 将抽象知识编译成可用的记忆条目
  └─ 操作选择: [flat_entries, hierarchical_entries, conditional_entries]
```

**搜索空间**: 每一层的操作选择 × 每一层的参数 = 记忆架构。

---

## 3. NAS-Inspired 解决方案

### 3.1 方案 A: Weight-Sharing Supernet 方法 (灵感: ENAS/DARTS)

**核心思想**: 不搜索整个记忆系统，而是维护一个"超级记忆管道"，通过学习路径权重来确定最优抽取策略。

```python
class MemorySupernet:
    """超级记忆管道 — 包含所有可能的抽取路径"""

    def __init__(self, dataset: list[DataInst]):
        # Layer 1: Instance-level extractors (并行执行所有)
        self.instance_extractors = [
            KeyFactExtractor(),        # 提取关键事实
            PatternExtractor(),        # 提取模式
            ErrorAnalyzer(),           # 分析常见错误
            ExampleSelector(),         # 选择代表性例子
        ]

        # Layer 2: Aggregation strategies (并行执行所有)
        self.aggregators = [
            TopicClusterer(),          # 按主题聚类
            DifficultyGrouper(),       # 按难度分组
            ErrorTypeClusterer(),      # 按错误类型聚类
        ]

        # Layer 3: Abstraction methods (并行执行所有)
        self.abstractors = [
            RuleInducer(),             # 归纳规则
            PatternSynthesizer(),      # 合成模式
            ExceptionCataloger(),      # 编目例外
        ]

        # 架构参数 α — 每条路径的权重
        self.alpha = {
            'layer1': softmax([1.0] * len(self.instance_extractors)),
            'layer2': softmax([1.0] * len(self.aggregators)),
            'layer3': softmax([1.0] * len(self.abstractors)),
        }

    def extract_memory(self, dataset, architecture_weights=None):
        """根据架构权重抽取记忆"""
        weights = architecture_weights or self.alpha

        # Layer 1: 加权混合所有 instance extractors 的输出
        instance_features = []
        for ext, w in zip(self.instance_extractors, weights['layer1']):
            if w > threshold:  # 剪枝低权重路径
                instance_features.append((w, ext.extract(dataset)))

        # Layer 2: 加权混合所有 aggregation 策略
        clusters = []
        for agg, w in zip(self.aggregators, weights['layer2']):
            if w > threshold:
                clusters.append((w, agg.aggregate(instance_features)))

        # Layer 3: 加权混合所有 abstraction 方法
        knowledge = []
        for abs_, w in zip(self.abstractors, weights['layer3']):
            if w > threshold:
                knowledge.append((w, abs_.abstract(clusters)))

        # 编译最终记忆
        return self.compile_memory(knowledge)
```

**优势**: 一次"训练"(在整个数据集上执行所有路径)，然后通过调整权重 α 来搜索最优架构，避免重复的全量评估。

**与 GEPA 的集成**:
- α 的优化可以用 GEPA 的进化循环
- 由于权重共享，改变 α 不需要重新抽取，只需重新组合
- Minibatch gate 现在有效了: 评估的是不同 α 配置下的同一组预计算特征

### 3.2 方案 B: Progressive Architecture Search (灵感: Progressive NAS)

**核心思想**: 从简单的记忆架构开始，逐步增加复杂度。

```
Stage 1: 单层记忆 (直接抽取)
  搜索空间: 只在 Layer 1 的操作中搜索
  评估成本: 低 (简单的抽取，快速评估)

  找到最优 Layer 1 配置 → 固定

Stage 2: 两层记忆 (抽取 + 聚合)
  搜索空间: 在 Layer 2 的操作中搜索 (Layer 1 已固定)
  评估成本: 中等

  找到最优 Layer 2 配置 → 固定

Stage 3: 三层记忆 (抽取 + 聚合 + 抽象)
  搜索空间: 在 Layer 3 的操作中搜索 (Layer 1, 2 已固定)
  评估成本: 较高但可控

Stage 4: 记忆编译优化
  搜索空间: 编译策略 (格式、组织方式)
  评估成本: 可用全量评估
```

**优势**: 每个 stage 的搜索空间很小，可以用小 budget 快速搜索。

### 3.3 方案 C: Block-wise Modular Optimization (灵感: DNA/HEP-NAS)

**核心思想**: 将记忆分成独立的 block，每个 block 对应数据集的一个子域，独立优化。

```python
class BlockwiseMemoryOptimizer:
    """分块记忆优化器"""

    def partition_dataset(self, dataset, n_blocks=5):
        """将数据集分成 n 个语义块"""
        # 使用 embedding 聚类
        embeddings = [embed(d['input']) for d in dataset]
        clusters = kmeans(embeddings, n_blocks)
        return {i: [d for d, c in zip(dataset, clusters) if c == i]
                for i in range(n_blocks)}

    def optimize_block(self, block_data, base_memory, block_id):
        """独立优化一个 block 的记忆"""
        # 只在 block_data 上评估 — 大幅减少评估成本
        # 使用标准 GEPA 进化循环，但只针对一个 block
        block_adapter = MemoryAdapter(
            evaluator=self.evaluator,
            base_system_prompt=self.base_prompt,
        )

        # 小数据集 → minibatch gate 有效!
        result = gepa.optimize(
            seed_candidate={"memory": base_memory},
            trainset=block_data,   # 只是 block 的数据
            valset=block_data,
            adapter=block_adapter,
            reflection_minibatch_size=3,  # 现在有效了!
            max_metric_calls=200,
        )
        return result.best_candidate

    def merge_blocks(self, block_memories):
        """合并所有 block 的记忆"""
        # 去重、冲突消解、全局一致性检查
        merged = deduplicate_entries(block_memories)
        return merged
```

**优势**:
- 每个 block 的数据量小(~数据集的 1/n)，minibatch gate 重新有效
- Block 间可以并行优化
- 自然地产生领域专业化的记忆条目

**与 Routing 的协同**: 这种 block-wise 结构天然地与 `RoutingMemoryAdapter` 配合:
- 每个 block 的记忆对应一组相关的 entry
- 路由器将查询定向到正确的 block
- 评估缓存效率最大化

### 3.4 方案 D: Proxy-Based Evaluation (灵感: Zero-Cost Proxies)

**核心思想**: 设计不需要完整评估的记忆质量代理指标。

```python
class MemoryQualityProxy:
    """记忆质量的快速代理评估"""

    def coverage_score(self, memory_entries, dataset):
        """覆盖率: 记忆条目覆盖了多少数据集的主题
        成本: O(|entries| × |dataset|) 的 embedding 相似度计算
        不需要 LLM 调用
        """
        entry_embeddings = [embed(e.content) for e in memory_entries]
        data_embeddings = [embed(d['input']) for d in dataset]

        covered = 0
        for d_emb in data_embeddings:
            max_sim = max(cosine_sim(d_emb, e_emb) for e_emb in entry_embeddings)
            if max_sim > THRESHOLD:
                covered += 1
        return covered / len(dataset)

    def specificity_score(self, memory_entries):
        """特异性: 记忆条目之间的差异度
        成本: O(|entries|^2)
        不需要 LLM 调用
        """
        embeddings = [embed(e.content) for e in memory_entries]
        pairwise_sims = [cosine_sim(a, b) for a, b in combinations(embeddings, 2)]
        return 1.0 - mean(pairwise_sims)  # 越不相似越好

    def information_density(self, memory_entries):
        """信息密度: 每个 entry 的信息量
        成本: O(|entries|)
        不需要 LLM 调用
        """
        return mean(len(e.content.split()) for e in memory_entries)

    def routing_entropy(self, memory_entries, dataset, router):
        """路由熵: 路由分布的均匀程度
        成本: O(|dataset|) 的路由调用 (可缓存)
        """
        route_counts = Counter()
        for d in dataset:
            routes = router.route(d['input'], memory_entries)
            for r in routes:
                route_counts[r] += 1
        # 计算熵 — 越均匀越好
        probs = [c / sum(route_counts.values()) for c in route_counts.values()]
        return -sum(p * log(p) for p in probs if p > 0)

    def composite_proxy(self, memory_entries, dataset):
        """组合代理分数 — 加权组合多个 proxy"""
        return (
            0.4 * self.coverage_score(memory_entries, dataset)
            + 0.3 * self.specificity_score(memory_entries)
            + 0.2 * self.routing_entropy(memory_entries, dataset, self.router)
            + 0.1 * self.information_density(memory_entries)
        )
```

**优势**: Proxy 评估不需要任何 LLM 推理调用，成本几乎为零。可以用来：
1. 快速筛选明显差的记忆变异 (替代 minibatch gate)
2. 引导搜索方向 (哪些 entry 需要改进)
3. 早停 (proxy 分数不再提升时)

---

## 4. 推荐的集成方案: Hierarchical Block-wise Search with Proxy Gate

综合上述方案的优点，推荐以下集成方案：

### 4.1 整体架构

```
Phase 0: Dataset Analysis & Blocking
  ├─ 使用 embedding 将数据集聚类为 k 个 block
  ├─ 识别每个 block 的主题特征
  └─ 输出: block_assignments, block_descriptions

Phase 1: Block-wise Memory Generation (Progressive)
  For each block b in parallel:
    ├─ Stage 1: 在 block_b 的数据上运行 GEPA
    │   ├─ minibatch_size = 3 (现在有效!)
    │   ├─ 评估集 = block_b 的数据 (较小)
    │   └─ 输出: block_b_memory
    └─ 使用 proxy score 快速评估

Phase 2: Cross-block Merge & Refinement
  ├─ 合并所有 block 的记忆
  ├─ 去重和冲突消解
  ├─ 在全量数据集上评估合并结果
  └─ 输出: merged_memory

Phase 3: Global Fine-tuning (Optional)
  ├─ 以 merged_memory 为种子
  ├─ 使用 Proxy Gate 替代 minibatch gate
  ├─ 在 proxy score 下降时拒绝变异
  └─ 定期全量验证 (每 N 次迭代)
```

### 4.2 关键创新点

#### 4.2.1 Proxy-based Acceptance Gate

替代 minibatch gate，使用 proxy 分数进行快速筛选：

```python
def proxy_gate(parent_memory, child_memory, dataset):
    """基于 proxy 的快速接受门控"""
    parent_proxy = composite_proxy(parent_memory, dataset)
    child_proxy = composite_proxy(child_memory, dataset)

    if child_proxy <= parent_proxy:
        return False  # 快速拒绝，成本 ≈ 0

    # Proxy 通过 → 使用 targeted evaluation
    # 只评估受影响的样本
    affected_samples = identify_affected_samples(parent_memory, child_memory, dataset)
    if len(affected_samples) == 0:
        return False

    # 在受影响样本上评估
    parent_score = evaluate(parent_memory, affected_samples)
    child_score = evaluate(child_memory, affected_samples)

    return child_score > parent_score
```

#### 4.2.2 Targeted Evaluation (关键效率提升)

不评估整个数据集，只评估**受变异影响的样本**:

```python
def identify_affected_samples(parent_memory, child_memory, dataset):
    """识别被变异影响的样本"""
    # 1. 找出变化的 entry
    changed_entries = diff_memories(parent_memory, child_memory)

    # 2. 使用 embedding 找出可能受影响的样本
    changed_embeddings = [embed(e.content) for e in changed_entries]
    data_embeddings = [embed(d['input']) for d in dataset]

    affected = []
    for i, d_emb in enumerate(data_embeddings):
        for c_emb in changed_embeddings:
            if cosine_sim(d_emb, c_emb) > RELEVANCE_THRESHOLD:
                affected.append(i)
                break

    return affected
```

**效率分析**: 如果变异改变了 1 个 entry，影响 5% 的数据集：
- 旧方案: 评估 500 个样本 → 500 次 LLM 调用
- 新方案: 评估 25 个样本 → 25 次 LLM 调用 (20x 加速)

#### 4.2.3 Cache-aware Block Evaluation

```python
class CacheAwareBlockEvaluator:
    """利用 RoutingMemoryAdapter 的缓存特性"""

    def evaluate_with_routing_cache(self, memory, changed_entry_keys):
        """只重新评估路由到变化 entry 的样本"""
        # 1. 对所有样本执行路由 (可缓存)
        routes = self.router.route_all(self.dataset, memory)

        # 2. 只有路由到 changed entries 的样本需要重新评估
        to_evaluate = [
            d for d, r in zip(self.dataset, routes)
            if any(key in changed_entry_keys for key in r)
        ]

        # 3. 其他样本的分数从缓存获取
        cached_scores = self.cache.get_batch(memory, unchanged_ids)

        return merge(cached_scores, self.evaluate(memory, to_evaluate))
```

### 4.3 记忆层级抽取架构的搜索空间定义

```python
# 搜索空间定义
MEMORY_SEARCH_SPACE = {
    # Layer 1: 如何从原始数据中提取信息
    "instance_extraction": {
        "method": ["verbatim", "key_facts", "patterns", "error_analysis"],
        "granularity": ["per_sample", "per_batch", "per_topic"],
        "max_tokens_per_item": [50, 100, 200, 500],
    },

    # Layer 2: 如何聚合提取的信息
    "aggregation": {
        "clustering_method": ["embedding_kmeans", "topic_model", "error_type"],
        "n_clusters": [3, 5, 10, 20],
        "aggregation_strategy": ["union", "intersection", "majority_vote"],
    },

    # Layer 3: 如何抽象聚合的信息
    "abstraction": {
        "method": ["rule_induction", "pattern_synthesis", "example_selection"],
        "abstraction_level": ["specific", "moderate", "general"],
        "max_rules": [5, 10, 20, 50],
    },

    # Layer 4: 如何组织最终记忆
    "compilation": {
        "format": ["flat_entries", "hierarchical", "conditional"],
        "max_entries": [10, 20, 50],
        "entry_max_length": [100, 200, 500, 1000],
    },
}
```

---

## 5. 与 NAS 技术的对应关系

### 5.1 直接可用的 NAS 技术

| NAS 技术 | 在 Memory Search 中的应用 | 实现难度 |
|---------|------------------------|---------|
| **Weight Sharing (ENAS)** | 不同记忆架构共享底层抽取结果 | 中 |
| **Progressive Search (PNAS)** | 从简单记忆逐步增加复杂度 | 低 |
| **Block-wise Modular (DNA)** | 按数据子域独立优化记忆块 | 低 |
| **Zero-cost Proxy** | 用 coverage/specificity 等指标替代全量评估 | 低 |
| **Predictor-based (Surrogate)** | 训练一个模型预测记忆系统的性能 | 高 |
| **DARTS (Differentiable)** | 连续松弛抽取操作的选择 | 高 |
| **Once-for-All** | 训练一个包含所有子架构的超记忆 | 高 |
| **Multi-objective Pareto** | 平衡记忆覆盖率、大小、推理成本 | 中 (已有) |

### 5.2 GEPA 已有基础设施的复用

| GEPA 组件 | 可复用于 |
|----------|--------|
| `EvaluationCache` | Block-wise 评估结果缓存 |
| `ParetoCandidateSelector` | 多目标记忆架构选择 |
| `AdaBoostBatchSampler` | 聚焦于记忆覆盖薄弱区域的采样 |
| `RoutingMemoryAdapter` | 天然支持 block-wise 评估和缓存 |
| `edit-based mutation` | 局部记忆变异操作 |

---

## 6. 文献基础

### 6.1 NAS 核心文献

1. **DARTS** (Liu et al., 2019) — 可微架构搜索，通过连续松弛将离散搜索空间转化为连续优化问题
2. **ENAS** (Pham et al., 2018) — 通过权重共享将 NAS 成本从数千 GPU 天降至数小时
3. **Progressive NAS** (Liu et al., 2018) — 从简单到复杂逐步搜索，每一步固定已搜索的部分
4. **Once-for-All** (Cai et al., 2020) — 训练一个超网络，部署时直接提取子网络
5. **HEP-NAS** (Li et al., 2024) — 层级式少样本切分，通过梯度匹配划分操作
6. **DNA Block-wise** (Wang et al., 2024) — 模块化超网络训练，通过 block-wise 蒸馏提升排序可靠性

### 6.2 NAS 效率技术

7. **Zero-Cost Proxies** (Mellor et al., 2020) — 单个 minibatch 前向传播估计架构性能
8. **TG-NAS** (2024) — 通用零代价代理，跨搜索空间泛化
9. **UP-NAS** (CVPR 2024) — 统一代理，同时预测多个代理分数
10. **LaMOO** (JMLR 2024) — 基于蒙特卡洛树搜索的多目标 NAS，比 BO 提升 200% 样本效率

### 6.3 多目标 NAS

11. **MTF-PDNS** (2024) — 基于 Pareto 支配的新颖性搜索，使用 training-free 指标
12. **EGBO** (2024) — 进化引导贝叶斯优化，集成选择压力与 qNEHVI
13. **Multi-Objective Differentiable NAS** (2024) — 编码用户偏好实现多目标平衡

### 6.4 Compound AI 系统优化 (与 GEPA 最直接相关)

14. **DSPy / MIPROv2** (Khattab et al., 2023; Opsahl-Ong et al., 2024) — 模块化 LM 程序优化框架。MIPROv2 使用三阶段策略：Bootstrapping (运行管道收集 traces) → Grounded Proposal (利用 traces+code+data 起草候选指令) → Discrete Search (贝叶斯优化 TPE 搜索指令×示例的组合空间)。**本质上是 LLM prompt 管道的 NAS**：搜索空间=所有模块的指令×示例组合，搜索策略=surrogate-based BO，评估=端到端管道性能。
15. **TextGrad** (Yuksekgonul et al., 2025) — "文本的 Autograd"，用 LLM 反馈作为梯度信号迭代优化文本输出。与 DSPy 互补：TextGrad 擅长 test-time refinement，DSPy 擅长 compile-time optimization。
16. **EvoPrompt** (Guo et al., ICLR 2024) — 将 LLM 与进化算法 (GA/DE) 结合做离散 prompt 优化。在 BIG-Bench Hard 上比手工 prompt 提升 25%。关键洞察：LLM 可以充当进化算法中的变异/交叉算子。
17. **Promptbreeder** (Fernando et al., ICML 2024) — 自指涉自改进：同时进化 task-prompt 和 mutation-prompt。元级别变异是关键创新——搜索策略本身在进化。类比 NAS 中"学习控制器"。
18. **TRIPLE** (NeurIPS 2024) — 将 prompt 选择建模为多臂老虎机的最优臂识别问题。使用自适应采样高效识别最优 prompt。比基线提升 3-16%。

### 6.5 LLM 记忆系统文献

19. **MemGPT/Letta** — 层级记忆管理系统，区分工作记忆和长期记忆
20. **RAPTOR** (Sarthi et al., 2024) — 递归抽象处理树，层级化文档总结和检索
21. **Self-RAG** (Asai et al., 2024) — 自适应检索增强生成
22. **GraphRAG** (Microsoft, 2024) — 基于知识图谱的检索增强

### 6.6 层级搜索空间

23. **Context-Free Grammar-Based Hierarchical NAS** (NeurIPS 2024) — 基于上下文无关文法的层级搜索空间统一框架，可生成比标准空间大 100+ 数量级的搜索空间。**关键启示**: 文法可以自然表达组合式、层级化的搜索空间结构，直接适用于记忆抽取管道的搜索空间定义。
24. **LaMOO** (JMLR 2024) — 学习搜索空间分区，聚焦于可能包含 Pareto 最优解的区域。比标准 BO 和进化方法提升 200%+ 样本效率。

### 6.7 GEPA 项目已有研究

25. **Shared Active Subset** — 共享信息性测试子集 (knowledge/ideas/)
26. **SPRT Adaptive Gate** — 序贯概率比检验替代固定 minibatch gate (knowledge/ideas/)
27. **Ensemble Evolution** — 集成感知的进化优化 (knowledge/ideas/)

---

## 7. 关键启示: DSPy MIPROv2 的方法论映射

NAS Agent 调研中发现的 **DSPy/MIPROv2** 框架与 GEPA Memory Adapter 的问题最为对口。MIPROv2 解决的正是"多模块 LLM 管道的组合优化"问题，其方法论可直接映射到记忆系统优化：

### 7.1 MIPROv2 三阶段策略 → 记忆系统的映射

| MIPROv2 阶段 | 记忆系统对应 | 实现方式 |
|-------------|------------|---------|
| **Stage 1: Bootstrapping** — 运行管道，收集 traces，按质量过滤 | **记忆种子生成** — 在整个数据集上运行，收集失败案例和成功模式 | 一次性前置成本，不在循环内 |
| **Stage 2: Grounded Proposal** — 用 traces+code+data 起草候选指令 | **记忆条目生成** — 基于失败模式和知识缺口提出候选记忆内容 | 类似现有的 reflective mutation |
| **Stage 3: Discrete Search** — TPE 贝叶斯优化搜索组合空间 | **记忆组合优化** — 搜索哪些 entry 的哪个版本组合效果最好 | **新增**: 用 BO 替代逐条编辑 |

### 7.2 核心洞察: 将记忆优化从"逐条编辑"重构为"组合搜索"

当前 GEPA MemoryAdapter 的范式:
```
循环: 选择一条 entry → edit → 在整个数据集上评估 → 接受/拒绝
问题: 每次只改一条，需要整个数据集评估，效率极低
```

借鉴 MIPROv2 的范式:
```
Phase 1 (离线): 在整个数据集上 bootstrap，生成 N 个候选 entry 版本池
Phase 2 (搜索): 用贝叶斯优化搜索 entry 版本的最优组合
  搜索空间: entry_1 ∈ {v1_a, v1_b, v1_c} × entry_2 ∈ {v2_a, v2_b} × ...
  每次评估: 在固定的代表性子集上评估组合效果
  优化器: TPE (Tree-Parzen Estimator) 或 GEPA 自己的进化搜索
```

**关键区别**: 不再是"每次改一条 entry 然后全量评估"，而是:
1. **前置成本**: 一次性生成候选 entry 版本池 (利用全数据集的信息)
2. **搜索成本**: 搜索最优组合 (每次评估只需代表性子集，因为是比较不同组合)
3. **组合空间**: $\prod_i |V_i|$ 个可能组合，用 BO/进化搜索高效探索

### 7.3 Promptbreeder 的元变异思想

Promptbreeder 同时进化 task-prompt 和 mutation-prompt。映射到记忆系统:

- **Task-prompt → Memory entries**: 直接给 LLM 使用的知识
- **Mutation-prompt → Extraction strategy**: 控制如何从数据集中提取知识

当前 `EDIT_PROPOSAL_PROMPT` 是固定的变异指令。如果我们同时进化这个指令本身，就能让系统学习"什么样的记忆提取策略对当前任务最有效"。

---

## 8. 实施路线图

### Phase 0: MIPROv2-style Bootstrap (前置阶段)

**目标**: 一次性生成高质量的候选记忆版本池

改动:
1. 在整个数据集上运行 seed candidate，收集所有 traces
2. 按错误类型聚类，识别知识缺口
3. 为每个缺口生成多个候选 entry 版本
4. 输出: `entry_version_pool = {entry_key: [version_1, version_2, ...]}`

**预期效果**: 避免在循环中反复全量评估，前置成本一次性付出

### Phase 1: Proxy Gate + Targeted Evaluation (最小可行方案)

**目标**: 让 Memory Adapter 的进化循环效率可接受

改动:
1. 实现 `MemoryQualityProxy` 类 (coverage, specificity, routing_entropy)
2. 在 `engine.py` 中添加 proxy-based gate 选项
3. 实现 `identify_affected_samples()` 用于 targeted evaluation
4. 修改 `RoutingMemoryAdapter` 利用路由信息加速评估

**预期效果**: 评估成本降低 10-20x

### Phase 2: Block-wise Optimization (核心效率提升)

**目标**: 使 minibatch gate 在记忆优化中重新有效

改动:
1. 实现数据集聚类 (embedding-based)
2. 实现 `BlockwiseMemoryOptimizer`
3. 实现 block 合并和去重逻辑
4. 与 `RoutingMemoryAdapter` 集成

**预期效果**: 每个 block 可以高效使用 minibatch gate

### Phase 3: Hierarchical Architecture Search (高级方案)

**目标**: 自动搜索最优的记忆抽取架构

改动:
1. 定义记忆抽取搜索空间
2. 实现 Progressive Search 策略
3. 实现 Weight-Sharing Supernet (可选)
4. 集成到 GEPA 的 `optimize()` API

**预期效果**: 自动发现比人工设计更好的记忆架构

---

## 9. 与现有 GEPA 组件的关系

### 8.1 MemoryAdapter 的改造方向

当前 `MemoryAdapter` 将记忆视为一个扁平的 XML key-value store，通过 edit-based mutation 逐步改进。这个设计在以下情况下工作：

- 数据集较小 (< 50 个样本)
- 记忆条目数量少 (< 10 个)
- 每个条目影响大部分样本

当数据集增大或记忆系统需要更精细的领域专业化时，需要引入本文档描述的架构搜索方法。

### 8.2 RoutingMemoryAdapter 是天然的桥梁

`RoutingMemoryAdapter` 已经实现了关键的基础设施:
- **Per-query routing**: 可以精确知道哪些 entry 被用于哪些查询
- **Deterministic routing**: 确保缓存一致性
- **Cache efficiency**: 变异只影响路由到变化 entry 的查询

在此基础上，Block-wise optimization 和 Targeted evaluation 可以自然地集成。

### 8.3 不需要改变的部分

- GEPA 的核心进化循环 (`engine.py`)
- Pareto 前沿管理 (`state.py`)
- 候选选择策略 (`candidate_selector.py`)
- 基础的 edit-based mutation 机制

这些组件在 block 级别可以直接复用。

---

## 10. 风险与缓解

| 风险 | 缓解策略 |
|------|---------|
| Block 划分质量差 | 使用多种聚类方法 + 人工验证 |
| Block 间知识依赖 | 在 merge 阶段处理交叉引用 |
| Proxy 分数与真实性能相关性低 | 定期全量验证校准 proxy 权重 |
| 搜索空间过大 | 用 Progressive Search 逐步展开 |
| LLM 抽取的不确定性 | 多次抽取 + 投票/聚合 |

---

## 11. 总结

Memory Adapter 的效率瓶颈本质上是**局部变异 vs 全局评估**的矛盾。NAS 领域在过去十年中发展出了丰富的技术来处理类似的搜索效率问题。

最有前景的三个方向是:

1. **Block-wise Modular Optimization**: 将大问题分解为小问题，让 GEPA 的 minibatch gate 重新有效
2. **Proxy-based Evaluation**: 用零成本代理指标替代昂贵的 LLM 评估
3. **Targeted Evaluation**: 利用变异的局部性，只评估受影响的样本

这三者可以组合使用，预期能将记忆系统优化的成本降低 10-50 倍，使其在实际应用中变得可行。
