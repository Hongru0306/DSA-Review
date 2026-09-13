## DSA-Review

Official implementation of our paper.

All relevant resources will be released upon acceptance.

---

# 超图 RAG 核心算法库

把散落在研究工程数百个脚本中的**核心算法**整理为干净、模块化的 Python 源码,便于
复用 / 复现 / 交付。**不含任何数据文件**,所有输入以 JSON dict / `list[str]` 传入。

覆盖范围(与论文口径一致):

| 模块 | 内容 |
|---|---|
| 构图 `graph` | 语料切片 → 实体 / 关系图(确定性无 LLM,或 LLM 增量缓存);查询实体 / 关系解析 |
| 检索 `retrieval` | **Ours** 超图子图检索(论文 Alg 1 + 2)+ 基线 GraphRAG-lite / HippoRAG-lite / LightRAG-lite / RAPTOR-lite / Dense BGE / BM25 |
| 生成 `generation` | 证据问答(证据→答案,v42 引用 / 长度校准 + 校验修复);条款归因审查(论文 Alg 3);prompts 集中 |
| 评测 `evaluation` | 检索 Hit/Recall/Precision/MAP/MRR/nDCG/覆盖率;生成 Char/Token F1、EM、ROUGE-L、BLEU1;修订 diff / BERTScore 风格嵌入 F1;聚合 |
| LLM `llm` | DeepSeek / Qwen-vLLM 统一客户端(thinking 关闭、JSON 模式、重试、JSON 修复) |

模块以**顶层包**形式导入(仓库根即源码根)。

## 安装 / 导入

```bash
pip install -r requirements.txt        # 或 pip install -e .
cd <repo>
python -c "from retrieval.ours import OursRetriever"
```

## 快速上手(离线演示)

```bash
PYTHONIOENCODING=utf-8 python examples/demo.py
```

合成 5 条款语料 → 确定性构图 → 跑全部方法 + 检索指标,无需数据 / 模型 / 网络。

## 测试

```bash
pytest -q
```

全部使用注入的假编码器与微型合成语料,免下载免联网;Ours 评分对照手工推导公式、
基线返回全序、指标按手算真值断言。

## 数据 Schema(输入契约)

- **语料(corpus)**:`List[str]`,每条是条款 / 语料切片文本。
- **corpus_graph(可选,离线构图产物)**:`Dict[str, Dict]`,键为
  `sha1(norm(text))`(见 `utils.text.doc_cache_key`),值为
  `{"entities": [...], "relations": [["A","B"], ...], "llm_ok": bool}`。
  用 `graph.build_corpus_graph(corpus)`(无 LLM)生成,或
  `graph.build_corpus_graph_cache(...)`(LLM)。
- **qcache(可选,查询侧 LLM 缓存)**:`Dict[str, Dict]`,
  `{question: {"entities": [...], "relations": [...]}}`;
  用 `graph.build_query_cache(questions, path, llm, concurrent)` 生成。
- **检索行(输出 / 生成输入)**:`{"method", "qid", "question", "gold": [doc_idx],
  "retrieved_top10": [corpus 行, 含 doc_id/spec_name/clause/text]}`。

## 检索用法示例

```python
from retrieval import OursRetriever, BM25
from graph import build_corpus_graph
from evaluation import retrieval_metrics

ours = OursRetriever(n=3, alpha=0.03, lambda_=0.85)
ours.build(corpus, encoder, corpus_graph=build_corpus_graph(corpus))   # encoder: 支持 encode_batch 的对象
ranking = ours.rank("养护时间不应少于14天")[:5]
print(retrieval_metrics(ranking, gold=[0], k=5))

bm25 = BM25();  bm25.build(corpus, encoder)
```

`encoder` 默认用 `encoder.SemanticEncoder`(BGE)。测试可注入确定性假编码器(见 `tests/conftest.py`)。
`OursRetrieverTorch` 是 Ours 的 torch 向量化评分(`.to_torch(device)` 后用)。

## Ours 打分与参数

论文公式(测试与实现严格对齐):

```
omega_j(u) = idf(u)^0.25 * gamma^hop(u)            # 实体节点权重,hop 为到子图中心的距离
N_e   = Sum_t  mean_top_N( M_Q[t,u] * omega_j(u) )  # 实体覆盖项(M_Q = Z_Q Z^T)
N_r   = Sum_rho max_(u,rho,v) (M_Q[a,u]omega + M_Q[b,v]omega) / 2   # 关系项
Score = [ max(N_e - alpha*|V_j|, 0) + lambda*N_r ] / max(D_e + lambda*D_r, 1e-9)
```

`paper_score_formula=False` 时为默认公式(关系不乘 `lambda`);分母对全部候选恒定,
两式排序等价。

参数三组对照(均已收录 `config.py`):

| 出处 | N | gamma | m | alpha | lambda |
|---|---|---|---|---|---|
| 论文最优(参数网格 ACC 80.17%) | 3 | 0.7 | 2 | 0.03 | 0.85 |
| 20260720 冻结图配置 | 2 | 0.7 | 2 | 0.03 | 1.00 |
| 实验代码旧默认(无 lambda) | 3 | 0.7 | 2 | 0.02 | —— |

结构参数:`exp_cap=300`(扩展节点上限)、`local_cap=512`(局部边上限)、
`global_edge_cap=5000`、`q_cap=10`。

## 代码来源(忠实移植)

下表"源文件"列为研究工程内的相对路径,供溯源;本仓库为其整理后的模块化版本。

| 本仓库 | 源文件 |
|---|---|
| `retrieval/ours.py` + `ours_torch.py` | `scripts/eval_rag_at5.py` `OursCoverageGeneric` L894–1114;`scripts/ours_torch_accel.py` |
| `retrieval/{dense,bm25,ppr,lightrag_lite,raptor}.py` | `scripts/eval_rag_at5.py` L314–504 |
| `graph/builder.py` | `scripts/build_graphrag_novel_context5_graph_local.py` |
| `graph/llm_builder.py` | `scripts/eval_rag_at5.py` L1172–1253 |
| `generation/qa.py` | `autotune/code/scripts/run_construction_qa_long_generation_v42.py`;`scripts/run_construction_generation_20pct_20260713.py` |
| `generation/review.py` | `scripts/run_construction_review_v2.py` L4440–4533、L3478–3476 |
| `evaluation/*` | `scripts/eval_rag_at5.py` L100–212;`scripts/score_and_emit_tables_20260713.py`;`scripts/run_construction_review_v2.py` L3189–3371 |
| `llm/client.py` | `scripts/run_construction_review_v2.py` L302–628;`scripts/evidence_gated_generate.py` L147–186 |

> 注:表格中的 GraphRAG / LightRAG 基线为官方实验所用的 **lite 检查点**实现
> (实验代码实际跑的就是这些);本仓库不封装需要外部安装包的官方 `graphrag` /
> `lightrag` 适配器。

## 明确不含

- 数据文件、图谱索引 pkl/jsonl(只定义 schema 与示例)。
- 标注 / 预览任务外的弱基线(Random/SpecMarker/Length 等 flagging 启发式)、
  LLM-aug(HyDE/query2doc/IRCoT/FLARE)、RRF 融合。
