"""Global constants and default parameters, aligned with the experiment code / paper.

Two sets of Ours defaults:
- ``DEFAULT_PAPER_PARAMS``: grid optimum of the paper's parameter sweep,
  (N, alpha, lambda) = (3, 0.03, 0.85).
- ``DEFAULT_FROZEN_REPUBLISH_PARAMS``: 20260720 frozen reproduction config,
  (N=2, alpha=0.03, lambda=1.0).

The experiment code's ``OursCoverageGeneric`` old defaults were m=2 / gamma=0.7 /
topk(N)=3 / alpha=0.02 / no lambda (i.e. lambda=1), which differ from both sets.
"""

from __future__ import annotations

# ---- Query / construction caps (same as scripts/eval_rag_at5.py) ----
MAX_Q = 16          # max query entities
MAX_REL = 6         # max query relations
MAX_DOC_ENT = 24    # max entities per clause (corpus doc)
MAX_DOC_REL = 16    # max relation pairs per clause

# ---- Ours retrieval structure caps (same as the run scripts) ----
EXP_CAP = 300          # max BFS expansion nodes per clause
LOCAL_CAP = 512        # max local edges per clause
GLOBAL_EDGE_CAP = 5000 # keep the first N global edges
Q_CAP = 10             # query entities truncated by length

# ---- Default output ----
DEFAULT_TOP_K = 5

DEFAULT_PAPER_PARAMS = {
    "n": 3,            # entity aggregation size (the paper grid's N)
    "m": 2,            # BFS expansion hops
    "gamma": 0.7,      # hierarchy decay
    "alpha": 0.03,     # subgraph size penalty
    "lambda_": 0.85,   # relation weight
    "top_k": DEFAULT_TOP_K,
    "idf_power": 0.25,
    "use_rel": True,
    "use_llm": False,
    "paper_score_formula": True,
    "exp_cap": EXP_CAP,
    "local_cap": LOCAL_CAP,
    "global_edge_cap": GLOBAL_EDGE_CAP,
    "q_cap": Q_CAP,
}

DEFAULT_FROZEN_REPUBLISH_PARAMS = {
    "n": 2,
    "m": 2,
    "gamma": 0.7,
    "alpha": 0.03,
    "lambda_": 1.0,
    "top_k": DEFAULT_TOP_K,
    "idf_power": 0.25,
    "use_rel": True,
    "use_llm": False,
    "paper_score_formula": True,
    "exp_cap": EXP_CAP,
    "local_cap": LOCAL_CAP,
    "global_edge_cap": GLOBAL_EDGE_CAP,
    "q_cap": Q_CAP,
}

# ---- Stop words (same as eval_rag_at5.py) ----
STOP_EN = {
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "with", "by",
    "from", "as", "at", "is", "are", "was", "were", "be", "been", "being", "that",
    "this", "these", "those", "which", "who", "what", "when", "where", "why", "how",
    "does", "did", "do", "can", "could", "would", "should", "has", "have", "had",
    "it", "its", "their", "than", "then", "also", "not", "only", "using", "use",
}
STOP_ZH = {
    "根据", "依据", "规定", "要求", "进行", "是否", "说明", "理由", "下列", "以下",
    "哪一", "作为", "应当", "需要", "符合", "国家", "通用", "规范", "工程", "项目",
    "施工", "设计", "单位", "条文", "措施", "能力", "条件", "结构", "材料",
    "什么", "如何", "怎么", "哪些", "哪种", "可以", "不能", "不得", "应该", "必须",
}