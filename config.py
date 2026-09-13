"""全局常量与默认参数,与实验代码 / 论文取值对齐。

二组 Ours 默认参数:
- ``DEFAULT_PAPER_PARAMS``: 论文参数敏感性网格最优 (N, alpha, lambda) = (3, 0.03, 0.85)。
- ``DEFAULT_FROZEN_REPUBLISH_PARAMS``: 20260720 冻结复现图配置 (N=2, alpha=0.03, lambda=1.0)。

实验代码 `OursCoverageGeneric` 的旧默认值为 m=2 / gamma=0.7 / topk(N)=3 /
alpha=0.02 / 无 lambda(即 lambda=1),与二者均不同,已在 README 注明。
"""

from __future__ import annotations

# ---- 查询 / 构图上限(与 scripts/eval_rag_at5.py 一致)----
MAX_Q = 16          # 查询实体最大数
MAX_REL = 6         # 查询关系最大数
MAX_DOC_ENT = 24    # 每条款(corpus doc)最大实体数
MAX_DOC_REL = 16    # 每条款关系对最大数

# ---- Ours 检索结构上限(与运行脚本一致)----
EXP_CAP = 300          # 每条条款 BFS 扩展节点上限
LOCAL_CAP = 512        # 每条款局部边上限
GLOBAL_EDGE_CAP = 5000 # 全局边取前 N 条
Q_CAP = 10             # 查询实体按长度截断上限

# ---- 默认输出 ----
DEFAULT_TOP_K = 5

DEFAULT_PAPER_PARAMS = {
    "n": 3,            # 实体聚合大小(论文参数网格的 N)
    "m": 2,            # BFS 扩展跳数
    "gamma": 0.7,      # 层级衰减
    "alpha": 0.03,     # 子图规模惩罚
    "lambda_": 0.85,   # 关系权重
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

# ---- 停用词(与 eval_rag_at5.py 一致)----
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