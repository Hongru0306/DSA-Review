"""generation —— 证据问答(QA)与条款归因审查(review)。"""

from generation.prompts import (
    ENTITY_PROMPT,
    REL_PROMPT,
    CORPUS_GRAPH_PROMPT,
    QA_SYSTEM,
    OURS_LENGTH_CALIBRATION,
    REFUSALS,
)
from generation.qa import (
    build_answer_prompt,
    generate_answer,
    repair_prediction,
    select_retrieval_quote,
    split_question,
    valid_answer,
)
from generation.review import (
    dual_generation_score_fields,
    evidence_block,
    extract_predicted_correct_sentence,
    generate_review,
    parse_has_error,
    review_generation_prompt,
)

__all__ = [
    "ENTITY_PROMPT", "REL_PROMPT", "CORPUS_GRAPH_PROMPT",
    "QA_SYSTEM", "OURS_LENGTH_CALIBRATION", "REFUSALS",
    "build_answer_prompt", "generate_answer", "repair_prediction",
    "select_retrieval_quote", "split_question", "valid_answer",
    "dual_generation_score_fields", "evidence_block",
    "extract_predicted_correct_sentence", "generate_review",
    "parse_has_error", "review_generation_prompt",
]