# 与 Gemini 讨论: Prompt 进化优化策略

> 讨论日期: 2026-02-01
> Session ID: 3bcaaa2b-a88c-49e9-b7ff-bdabd0de4c53
> 基于: adaboost_candidate_analysis.md 的研究发现

---

## 讨论背景

基于 AdaBoost 实验的分析发现了以下关键问题：
1. 辛普森悖论：全样本正相关，后期负相关
2. 从最优父代变异不一定产生更好结果
3. 父代长度与子代改进呈负相关 (r = -0.184)
4. 失败变异倾向于"重组结构"，成功变异倾向于"累加内容"

---

## 第一轮讨论

### 我的问题
1. 怎么看"复杂度陷阱"现象？是否在其他进化算法中常见？
2. 如何改进系统？如何选择父代？如何设计变异策略？
3. 对 prompt engineering 有什么启示？

### Gemini 的回答

#### 关于复杂度陷阱：这是经典的"膨胀 (Bloat)"问题

在遗传规划（GP）中非常常见：
- **代码膨胀 vs. 收益递减**：长 Prompt 变得脆弱（Brittle），任何变异都可能破坏内部复杂的依赖关系
- **局部最优的物理含义**：候选者 #5 (0.8分) 像站在一个极尖的山峰上，往任何方向走都是下坡路

#### 系统改进建议

**A. 父代选择策略**
1. **引入"长度惩罚" (Parsimony Pressure)**
   - 公式：$Fitness = Score - \alpha \times \log(Length)$
2. **多目标优化**：使用帕累托前沿选择
3. **适应度共享**：降低相似候选者的选中概率

**B. 变异策略优化**
1. **结构保护**：
   - 核心指令锁定（"step-by-step", "check units"）
   - 模块化变异：分为 [Role], [Constraints], [Knowledge], [Reasoning]
2. **引入"剪枝"操作**：缩短臃肿 Prompt，恢复可变异性
3. **增量式变异**：以 Append 方式加入知识，而非重写

**C. Exploration vs Exploitation**
- 早期：鼓励大幅度结构重组
- 后期：切换到微调模式

#### 对 Prompt Engineering 的启示

1. **加尔定律 (Gall's Law)**：复杂系统从简单系统发展而来
2. **指令 > 知识**：推理逻辑比领域知识更重要
3. **过拟合是最大敌人**：长 Prompt 可能泛化能力差

---

## 第二轮讨论

### 我的追问

1. 关于剪枝：#5 本身不长（5.3k），子代反而更长（7k-10k），如何设计剪枝？
2. 变异的高方差（同一父代产生 0.4-0.733 的子代），应该增加次数还是设计更稳定的策略？
3. AdaBoost 采样导致过拟合？如何平衡解决难题和保持泛化性？
4. 如何定义 prompt 的"相似度"来保持多样性？

### Gemini 的回答

#### 关于剪枝与"变异膨胀"

**诊断**：LLM 在"优化"时有"解释性冗余"倾向，增加解释而非修改逻辑

**策略**：
1. **修改变异指令**：
   - 明确要求"保持长度尽量不变"
   - 引入"编辑器"角色，输出 Diff 而非完整 Prompt
2. **实施"密度检查"过滤**：
   - 指标：$\Delta Score / \Delta Length$
   - 硬约束：`Max_Length = Parent_Length * 1.2`
3. **两阶段变异**：
   - 阶段 1：生成（允许变长）
   - 阶段 2：压缩（专门 LLM 调用重写得更紧凑）

#### 关于变异的高方差

1. **降低 Temperature**：0.2-0.4，需要逻辑修正而非创意写作
2. **结构化变异**：
   - `System Role`: 保持不变
   - `Core Instructions`: **锁定保护**
   - `Knowledge Base`: **只允许 Append**
   - `Negative Constraints`: 允许修改
3. **"预选赛"过滤**：先跑 5 个典型题，不过就淘汰

#### 关于 AdaBoost 过拟合

**问题**：AdaBoost 用于组合弱分类器，但 Prompt 优化需要单体最优

**策略**：
1. **混合采样 (Replay Buffer)**：
   - Evaluation Set = 50% 错题 + 50% 随机正确题（作为 Anchor）
2. **剪裁权重**：单题权重上限 = 平均权重 × 3
3. **分离策略**：某些题可能需要不同方法

#### 关于多样性度量

1. **文本距离**：❌ 不推荐，文本差异不等于逻辑差异
2. **语义嵌入**：✓ 可行，Cosine Similarity > 0.95 视为同类
3. **行为向量 (Performance Vector)**：⭐ 最推荐
   - 定义 $V = [r_1, r_2, ..., r_n]$，$r_i$ 是第 i 题结果
   - 如果 Hamming Distance 很小，说明行为同质
   - 保留那些"错得不一样"的候选者

---

## 第三轮讨论

### 我的追问

1. 关于 0% 正确率的题目（ID 4, 29）：应该放弃还是重点关注？
2. Reflection + Mutation 流程如何改进？考虑"结构保护"和"累加而非重写"

### Gemini 的回答

#### 关于"全军覆没"的题目：建议"暂时冻结"

**为什么是"黑洞"**：
- 权重指数级上升
- 可能超出模型能力边界
- 过拟合风险：极端变异破坏其他题目逻辑

**建议策略：课程学习 (Curriculum Learning)**
- **阈值过滤**：如果前 N 轮正确率始终为 0%，标记为 OOD
- **操作**：权重归零或极低，专注于边缘题目
- **后期再战**：系统成熟后再单独开启攻坚战

#### 改进 Reflection + Mutation：外科手术式流程

**1. 模块化 Prompt 架构**
```markdown
<system_role> You are a physicist... </system_role>
<core_reasoning> Always think step-by-step... </core_reasoning> <!-- 核心保护区 -->
<knowledge_base> ... </knowledge_base> <!-- 增量累加区 -->
<constraints> ... </constraints> <!-- 易变异区 -->
```

**2. 改进后的流程**

**Step 1: 结构化诊断**
- Prompt: "错误是因为：A) 缺知识？B) 逻辑错？C) 格式错？"
- 目的: 分类问题，决定手术方案

**Step 2: 制定手术方案 (The Patch Plan)**
- 输出 JSON Patch 而非完整 Prompt：
```json
{
  "action": "append",
  "target_section": "knowledge_base",
  "content": "- Note: In quantum harmonic oscillators..."
}
```

**Step 3: 确定性组装**
- 使用 Python 代码执行修改，而非 LLM
- 100% 保证没动的地方不会变

**Step 4: 回归测试 (Sanity Check)**
- 全量评估前先跑最容易的题
- 做错则直接丢弃

---

## 核心结论与行动建议

### 关于父代选择
- ❌ 不要只从最优变异（贪婪策略）
- ✓ 保持多样性，关注"潜力股"（中等分数、短长度）
- ✓ 使用行为向量（Performance Vector）度量相似度

### 关于变异策略
- ❌ 不要让 LLM 重写整个 Prompt
- ✓ 模块化分区，锁定核心骨架
- ✓ 让 LLM 输出 Patch/Diff，Python 执行组装
- ✓ 限制长度增长（Max = Parent × 1.2）

### 关于评估策略
- ❌ 不要死磕 0% 正确率的题目
- ✓ 使用混合采样（错题 + 正确题 Anchor）
- ✓ 引入"预选赛"快速淘汰
- ✓ 课程学习，先易后难

### 新假设：复杂度陷阱
- 父代长度与子代改进呈负相关 (r = -0.184)
- 短父代的高分子代率 (46%) > 长父代 (33%)
- 最优候选者可能已过拟合，任何修改都是噪声

### 架构建议：外科手术式流程
```
[错题分析] → [结构化诊断] → [生成 Patch] → [Python 组装] → [回归测试] → [全量评估]
```

---

## 后续实验方向

1. 实现模块化 Prompt 架构（XML 分区）
2. 实现 Patch-based 变异策略
3. 引入长度惩罚到 Fitness 函数
4. 实现基于 Performance Vector 的多样性保持
5. 实现课程学习：冻结极端难题
