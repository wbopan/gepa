# Knowledge Base

Organized research notes, findings, and references for the GEPA project.

## architecture/

How GEPA works internally - algorithm design and codebase mapping.

- **gepa_algorithm_and_codebase_mapping.md** - Complete mapping of GEPA algorithm (arXiv:2507.19457) to codebase implementation
- **trainset_vs_valset.md** - Design rationale for the two-dataset approach (trainset for mutation feedback, valset for Pareto tracking)
- **wandb_tables_and_metrics.md** - Reference guide for W&B logging structure and metric namespaces

## findings/

Empirical results and statistical analysis from GEPA experiment runs.

- **adaboost_candidate_analysis.md** - Statistical analysis of 37 candidates: Simpson's Paradox, complexity trap, evolutionary paths
- **adaboost_max3_vs_max_analysis.md** - Why power=3 overweighting failed: gravity well effect and lineage diversity analysis
- **ensemble_statistical_analysis.md** - Oracle ensemble potential (+13.3%), why majority voting fails, routing vs voting
- **selector_optimization_potential_analysis.md** - Parent selection ROI analysis: fertility cliff, base camp sweet spot (0.6-0.7)

## ideas/

Proposals and brainstorming for future improvements.

- **sprt_adaptive_minibatch_gate.md** - Replace fixed minibatch gate with SPRT for ~57% evaluation savings
- **gemini_discussion_ensemble_evolution.md** - Four-phase roadmap for ensemble-aware optimization (Router-based MoE)
- **gemini_discussion_prompt_evolution.md** - Bloat mitigation, surgical mutation, patch-based approach
- **frontier_targeting_via_variance.md** - Frontier sample identification using variance-based scoring
- **shared_active_subset_coevolution.md** - Shared exam subset framework from coevolutionary literature

## literature/

Paper surveys and annotated bibliographies.

- **diversity_optimization_in_evolutionary_algorithms.md** - Survey of diversity methods: MAP-Elites, QDAIF, NSGA-III, Island Models
- **papers_on_diverse_expert_ensembles.md** - Curated catalog of 20+ papers on diverse expert sets with implementation patterns
