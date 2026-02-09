# WandB Tables and Metrics in GEPA

Project: `bmpixel/gepa-boost`

## Logged Tables (table-file)

| Table Name | Description | When Logged |
|------------|-------------|-------------|
| **candidate_prompts** | Text content of candidate prompts generated during evolution | Each new candidate |
| **minibatch_outputs** | Outputs/predictions from evaluating candidates on each minibatch | Each iteration |
| **train_sample_weights** | Sample weights for weighted sampling (AdaBoost-style) | Each iteration (weighted runs) |
| **valset_outputs** | Outputs/predictions from evaluating new candidates on validation set | Each new candidate |
| **candidate_outputs** | Similar to minibatch_outputs (older format, used in `pmax` run) | Deprecated |

## Metrics Reference

### Candidate Tracking (`candidate/`)

| Metric | Type | Description |
|--------|------|-------------|
| `candidate/best_idx` | int (monotonic) | Index of the current best candidate |
| `candidate/count` | int (monotonic) | Total number of candidates generated |
| `candidate/new_idx` | int (monotonic) | Index of the newly generated candidate |
| `candidate/selected_idx` | int | Index of the parent candidate selected for mutation |

### Pareto Selection (`pareto/`)

| Metric | Type | Description |
|--------|------|-------------|
| `pareto/val_agg` | float (monotonic) | Aggregated validation score across all objectives |
| `pareto/linear_best_idx` | int (monotonic) | Best candidate index according to linear Pareto scoring |
| `pareto/val_candidates.{0-N}` | list[float] | Per-sample validation scores for each candidate |

### Training Metrics (`train/`)

| Metric | Type | Description |
|--------|------|-------------|
| `train/batch_score_before` | int | Score on minibatch before accepting a candidate |
| `train/batch_score_after` | int | Score on minibatch after accepting a candidate |
| `train/batch_weight_avg` | float | Average weight of samples in current minibatch |
| `train/weight_min` | float | Minimum sample weight |
| `train/weight_max` | float | Maximum sample weight |
| `train/weight_avg` | float | Average sample weight (normalized to 1.0) |

### Validation Metrics (`val/`)

| Metric | Type | Description |
|--------|------|-------------|
| `val/best_agg` | float (monotonic) | Best aggregated validation score achieved |
| `val/new_score` | float | Validation score of newly generated candidate |
| `val/eval_count` | int (monotonic) | Number of validation samples evaluated |
| `val/total` | int (monotonic) | Total validation set size |

### Progress Tracking

| Metric | Type | Description |
|--------|------|-------------|
| `gepa_iteration` | int (monotonic) | Main iteration counter for GEPA loop |

### System Metrics (`system/`)

Automatically logged by wandb:

| Metric | Description |
|--------|-------------|
| `system/cpu` | CPU utilization |
| `system/memory.*` | Memory usage |
| `system/gpu.*` | GPU utilization, frequency, temperature, power |
| `system/disk.*` | Disk I/O and usage |
| `system/network.*` | Network send/receive |

## Experiment Runs

| Run Name | Strategy | Status | Key Result |
|----------|----------|--------|------------|
| `adaboost-rerun` | AdaBoost weighted | running | - |
| `adaboost-max3` | AdaBoost + max selector | finished | val_agg: 0.93 |
| `adaboost-max` | AdaBoost + max selector | finished | val_agg: 0.87 |
| `adaboost-avg` | AdaBoost + avg selector | finished | val_agg: 0.80 |
| `adaboost` | AdaBoost baseline | finished | val_agg: 0.93 |
| `baseline` | Uniform sampling | finished | val_agg: 0.90 |
| `bayesian` | Bayesian weighting | finished | val_agg: 0.97 |
| `pmax` | Max selector (old) | finished | val_agg: 0.83 |

## Querying with MCP Tools

```python
# Get project info
query_wandb_tool(
    query="query { project(name: \"gepa-boost\", entityName: \"bmpixel\") { runCount } }",
    variables={}
)

# Get run metrics
query_wandb_tool(
    query="""query RunsBasic($entity: String!, $project: String!) {
        project(name: $project, entityName: $entity) {
            runs(first: 10, order: "-createdAt") {
                edges { node { name displayName state historyKeys } }
                pageInfo { endCursor hasNextPage }
            }
        }
    }""",
    variables={"entity": "bmpixel", "project": "gepa-boost"}
)
```
