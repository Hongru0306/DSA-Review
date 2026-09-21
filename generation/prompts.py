"""All prompt templates centralized here, ported verbatim from the experiment scripts.

Source line numbers are noted per section; QA / review templates are referenced
in the generation modules.
"""

from __future__ import annotations

# ---- Graph construction / query entity-relation (scripts/eval_rag_at5.py L1117-1140) ----

ENTITY_PROMPT = """Extract key technical entities from the question.
Return at most {max_terms} short noun phrases, one per line. Do not number them.

Question:
{question}"""

REL_PROMPT = """Extract up to {max_rels} semantic relation pairs from the question.
Each line must be: entity A | entity B
Use short noun phrases. Return "None" if no clear relation exists.

Question:
{question}"""

CORPUS_GRAPH_PROMPT = """Extract a compact knowledge graph from the text.
Return JSON only with this schema:
{{"entities":["short noun phrase"],"relations":[["entity A","entity B"]]}}
Rules:
- entities: at most {max_terms}, important domain concepts only.
- relations: at most {max_rels}, only pairs where both entities appear in entities.
- use concise surface names from the text.

Text:
{text}"""

# ---- QA evidence answering (autotune/.../run_construction_qa_long_generation_v42.py build_prompt) ----

QA_SYSTEM = "严谨回答建筑规范问题；不输出思考过程，不拒答。"

REFUSALS = ("证据不足", "无法回答", "无法确定", "未提供", "不能确定", "不足以")

EVIDENCE_RULE_1DOC = """3. 只选择能直接回答问题的1条条文片段。不要引用定义、条文说明、相邻条文或背景条件，除非该片段本身包含最终答案。
4. 条文片段必须保留最终数值、公式、限值、适用条件或处置要求；不要整条搬运。片段通常35～100字。
5. 选证据时按“含有问题所问对象和最终答案”的标准选择；没有直接支撑最终答案的条文不要引用。
6. 如果多个片段给出不同方案、数值或条件，优先采用同时包含题目限定条件和最终答案词的主条文、主表格或强制性表述；不要用注释、条文说明、相邻条文或例外条件替代主结论，除非问题明确询问该注释或例外。"""
LENGTH_RULE_1DOC = "12. 控制长度：整段答案通常80～180字；优先贴近“根据“……”明确……”的短答案格式。"

EVIDENCE_RULE_2DOC = """3. 优先选择能直接回答问题的条文片段，通常1～2条：如果一条已经包含最终答案，就只用这一条；如果需要适用条件或定义，再补1条必要前提。不要为了凑数量引用无关条文。
4. 每条证据只摘录连续关键原文，必须保留最终数值、公式、限值、适用条件或处置要求；不要整条搬运。每个片段通常35～100字。
5. 选证据时按“含有问题所问对象和最终答案 > 含有必要适用条件或定义”的顺序。没有直接支撑最终答案的条文不要引用。
6. 如果多个片段给出不同方案、数值或条件，优先采用同时包含题目限定条件和最终答案词的主条文、主表格或强制性表述；不要用注释、条文说明、相邻条文或例外条件替代主结论，除非问题明确询问该注释或例外。"""
LENGTH_RULE_2DOC = "12. 控制长度：整段答案通常90～220字；优先贴近“根据“……”明确……”的短答案格式。"

EVIDENCE_RULE_MULTIDOC = """3. 多跳题通常选择2～3条证据：前面引用必要适用条件、定义或工序背景，最后一条必须包含最终答案词、最终数值、最终公式或最终处置要求。
4. 每条证据只摘录连续关键原文，必须保留最终数值、公式、限值、适用条件或处置要求；不要整条搬运。每个片段通常35～110字。
5. 选证据时按“含有问题所问对象和最终答案 > 含有适用条件或定义 > 同一构件/同一工序/同一验算的必要前提”的顺序。没有直接支撑最终答案的条文不要引用。
6. 如果多个片段给出不同方案、数值或条件，优先采用同时包含题目限定条件和最终答案词的主条文、主表格或强制性表述；不要用注释、条文说明、相邻条文或例外条件替代主结论，除非问题明确询问该注释或例外。"""
LENGTH_RULE_MULTIDOC = "12. 控制长度：整段答案通常120～280字；优先贴近“根据“……”明确……”的短答案格式。"

V42_ANSWER_TEMPLATE = """你是建筑工程规范审查问答助手。请只根据给定的检索结果回答问题。

**严格输出格式**（必须遵守）：
```
根据"<连续原文片段>"["<必要前置条件原文>"]，明确<主体><标准短答案>。
```

**规则**：
1. **第一段必须 verbatim 引用包含最终答案的条文**——含数值、公式、限值、等级或判断词。
2. **第二段（可选）只引用真正必要的前置条件**——仅当一句话就能解释问题时才用。
3. **结论必须包含问题主体 + 标准答案**——例如"明确x0=0.3hb"、"明确不应低于4.0MPa"、"明确湿喷工艺"。
4. **禁用规范名称、编号、条款号**——不要写"GB50010《混凝土结构设计规范》第4.1.7条"。
5. **公式按输入原文逐字复制**——若检索结果是纯文本公式（如"x0=0.3hb"），照写；若输入包含LaTeX，去掉"$$"符号后照写。
6. **必须给出答案**——基于 top{topk} 中最相关的原文片段做保守回答。如果没有任何相关片段，引用 top1 第一句作为依据。
7. **总长度 180-220 字**：引文片段必须完整复制必要的原文，不为了凑长度加废话。
8. 不输出 Markdown、列表、"结论："等格式。

{extra_instruction}

【上下文背景】
{context}

【问题】
{query}

【检索结果】
{chunks}
"""

OURS_LENGTH_CALIBRATION = """【Ours统一长度校准】
覆盖上面的180–220字要求：最终答案控制在125–170个Unicode字符。
通常只逐字引用1个最相关且含最终数值、公式、等级或判断词的连续原文片段；只有结论确实依赖独立前置条件时才增加第2个片段。
引文后用最短句复述问题主体和直接答案，不添加通用背景、规范沿革或与问题无关的执行建议。"""

# ---- 抽取式 QA(scripts/evidence_gated_generate.py L124-144)----

EVIDENCE_GATED_SYSTEM = "You are a strict extractive QA system. Do not infer facts that are not explicitly supported."

EVIDENCE_GATED_USER_TEMPLATE = """{no_think}You are a construction regulation expert. Answer the question based on the retrieved evidence below.
Be concise and direct. If the question is multiple-choice, output only the letter (e.g. A, B, C or D).
For open-ended questions, give a short factual answer based strictly on the evidence.
Do not output any reasoning, analysis, or thinking process.

Retrieved evidence:
{evidence}

Question:
{question}

Answer:"""

# ---- 条款归因审查(scripts/run_construction_review_v2.py review_generation_prompt)----

REVIEW_SYSTEM = "你是严格依据检索证据进行施工规范审查的回答API。"

REVIEW_PROMPT_PRE_GATE = """你是建筑工程规范审查问答助手。请只根据给定的检索结果审核并纠正待审核句。

要求：
1. 回答采用“检索条文原文复述 + 审查结论”的形式。关键规范依据必须逐字摘录，不要用外部知识补充检索结果中不存在的具体要求。
2. 只允许依据给定top10检索文档和施工上下文作答。关键数值、公式、等级、构件名、施工措施及验收判定，必须能在所引用的原文片段中找到。
3. 必须先识别上下文中的工程对象、工况和前置条件，再把相关条文按依赖顺序连接起来；若结论需要多条证据，不得只引用其中一条。
4. 每条证据只摘录连续关键原文，保留适用条件、限值、程序或处置要求，不要整条搬运。
5. 不得因为待审核句看起来安全、保守或语法通顺就直接判定；必须完成证据推导。
6. 即使证据不完整，也不得输出“无法确定”“无法回答”“未提供相关条款”等拒答表述，应基于最相关检索结果给出保守明确的审核意见。
7. 如果待审核句有错误，应给出一条可直接替换原句的完整修改句；如果没有错误，明确写“无错误，无需修改”，不得为了改写而改写。
8. 不使用Markdown列表或加粗。严格输出以下两个部分，每个部分一段：
问题原因：先引用必要证据，再说明这些证据如何结合上下文推出审核结论。
修改建议：有错误。建议改为：“完整句子” / 无错误，无需修改。
{scenario_note}

【施工上下文】
{scenario_context}

【待审核句】
{review_sentence}

【检索结果】
{evidence_block}"""

REVIEW_PROMPT_GATE = """你是建筑工程规范审查问答助手。请只根据给定的检索结果审核并纠正待审核句。

要求：
1. 回答采用“检索条文原文复述 + 审查结论”的形式。关键规范依据必须逐字摘录，不要用外部知识补充检索结果中不存在的具体要求。
2. 只允许依据给定top10检索文档和施工上下文作答。关键数值、公式、等级、构件名、施工措施及验收判定，必须能在所引用的原文片段中找到。
3. 必须先识别上下文中的工程对象、工况和前置条件，再把相关条文按依赖顺序连接起来；若结论需要多条证据，不得只引用其中一条。
4. 每条证据只摘录连续关键原文，保留适用条件、限值、程序或处置要求，不要整条搬运。
5. 先把待审核句的每个实质要求分别标为“被证据直接支持”“与证据直接冲突”或“检索结果未覆盖”。只有至少一项与适用条文直接冲突时，才判定“有错误”。
6. “直接冲突”是指同一工程对象及适用条件下，数值或单位、比较关系、材料或方法、工序或程序、允许或禁止关系互不相容。仅仅没有检索到逐字相同表述、证据未覆盖某一细节，不能推定为错误。
7. 同义表达、简称、语序差异、无害的具体化、未写规范编号以及不改变技术要求的措辞差异，均不构成错误；不得为语言润色而修改。
8. 若证据不足以证明直接冲突，不得输出“无法确定”等拒答表述，应判定“无错误，无需修改”，并说明现有证据未形成反证。
9. 如果存在直接冲突，应给出一条可直接替换原句的完整修改句；如果没有直接冲突，必须原句保留并明确写“无错误，无需修改”。
10. 不使用Markdown列表或加粗。严格输出以下两个部分，每个部分一段：
问题原因：先引用必要证据，再说明这些证据如何结合上下文推出审核结论。
修改建议：有错误。建议改为：“完整句子” / 无错误，无需修改。
{scenario_note}

【施工上下文】
{scenario_context}

【待审核句】
{review_sentence}

【检索结果】
{evidence_block}"""

REVIEW_SCENARIO_NOTE_PRE = (
    "9. 每条检索证据后的“本次查询匹配的条文场景词”来自该条文上下文图，仅用于识别适用场景；"
    "它不是独立规范结论，具体要求仍必须由条文原文支持。\n"
)
REVIEW_SCENARIO_NOTE = (
    "11. 每条检索证据后的“本次查询匹配的条文场景词”来自该条文上下文图，仅用于识别适用场景；"
    "它不是独立规范结论，具体要求仍必须由条文原文支持。\n"
)

REVIEW_RETRY_INSTRUCTION = "上次未严格输出“问题原因：”和“修改建议：”两个部分，请按指定格式重新回答。"