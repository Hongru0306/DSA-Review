"""Revision / edit / embedding evaluation metrics (construction review pipeline).

Port of ``scripts/run_construction_review_v2.py``:
- ``revision_edit_fragments``: extract edit fragments relative to the original
  sentence with ``difflib.SequenceMatcher`` (Ratcliff-Obershelp, not Levenshtein).
- ``revision_diff_char_prf``: character PRF over the edit payload, avoiding the
  inflation from copying the whole sentence.
- ``embedding_prf``: BERTScore-style -- BGE chunk cosine, clipped to [0,1].
- ``delta_embedding_prf``: cosine of the edit-direction embedding difference
  times the length ratio.
"""

from __future__ import annotations

import collections
import difflib
import re

import numpy as np

from utils.text import normalized_chars, normalize_space


def revision_char_prf(prediction, reference) -> tuple[float, float, float]:
    """Multiset PRF over digits / latin / CJK only (review-pipeline convention)."""
    pred = normalized_chars(prediction)
    gold = normalized_chars(reference)
    if not pred or not gold:
        return 0.0, 0.0, 0.0
    common = sum((collections.Counter(pred) & collections.Counter(gold)).values())
    precision = common / len(pred)
    recall = common / len(gold)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def revision_edit_fragments(original, revised) -> list[str]:
    """Return only the text introduced relative to the original sentence (replace / insert / explicit delete).

    Copied context is excluded, so that copying a large span cannot earn a high
    revision F1.
    """
    source = normalize_space(original)
    target = normalize_space(revised)
    matcher = difflib.SequenceMatcher(None, source, target, autojunk=False)
    fragments: list[str] = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        old = source[i1:i2].strip(" \t\r\n，,；;。.!！?？:：")
        new = target[j1:j2].strip(" \t\r\n，,；;。.!！?？:：")
        fragment = new if new else (f"删除{old}" if old else "")
        if normalized_chars(fragment):
            fragments.append(fragment)
    return fragments


def revision_diff_char_prf(
    original,
    prediction,
    reference,
) -> tuple[float, float, float, list[str], list[str]]:
    """Character PRF over the edit payload (main revision-evaluation convention)."""
    predicted_fragments = revision_edit_fragments(original, prediction)
    reference_fragments = revision_edit_fragments(original, reference)
    if not predicted_fragments and not reference_fragments:
        return 1.0, 1.0, 1.0, predicted_fragments, reference_fragments
    if not predicted_fragments or not reference_fragments:
        return 0.0, 0.0, 0.0, predicted_fragments, reference_fragments
    precision, recall, f1 = revision_char_prf(
        "；".join(predicted_fragments),
        "；".join(reference_fragments),
    )
    return precision, recall, f1, predicted_fragments, reference_fragments


def revision_embedding_chunks(text) -> list[str]:
    """Split a complete revised sentence into semantic units (by sentence, dropping empty pieces)."""
    value = normalize_space(text)
    chunks = [
        chunk.strip(" \t\r\n，,；;。.!！?？:：")
        for chunk in re.split(r"[；;。.!！?？\r\n]+", value)
    ]
    chunks = [chunk for chunk in chunks if normalized_chars(chunk)]
    return chunks or ([value] if normalized_chars(value) else [])


def embedding_prf(
    prediction_chunks: list[str],
    reference_chunks: list[str],
    embeddings: dict[str, np.ndarray],
) -> tuple[float, float, float]:
    """BERTScore style: full-revision chunked BGE embedding cosine, clipped to [0,1]."""
    if not prediction_chunks and not reference_chunks:
        return 1.0, 1.0, 1.0
    if not prediction_chunks or not reference_chunks:
        return 0.0, 0.0, 0.0
    pred_matrix = np.stack([embeddings[text] for text in prediction_chunks])
    gold_matrix = np.stack([embeddings[text] for text in reference_chunks])
    similarities = np.clip(pred_matrix @ gold_matrix.T, 0.0, 1.0)
    precision = float(np.mean(np.max(similarities, axis=1)))
    recall = float(np.mean(np.max(similarities, axis=0)))
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def delta_embedding_prf(
    original,
    prediction,
    reference,
    embeddings: dict[str, np.ndarray],
) -> tuple[float, float, float]:
    """Compare the semantic direction and magnitude of the edit: ``embedding(revised) - embedding(original)``."""
    source = normalize_space(original)
    predicted = normalize_space(prediction)
    gold = normalize_space(reference)
    if not source or not predicted or not gold:
        return 0.0, 0.0, 0.0
    pred_unchanged = normalized_chars(predicted) == normalized_chars(source)
    gold_unchanged = normalized_chars(gold) == normalized_chars(source)
    if pred_unchanged and gold_unchanged:
        return 1.0, 1.0, 1.0
    if pred_unchanged or gold_unchanged:
        return 0.0, 0.0, 0.0

    pred_delta = embeddings[predicted] - embeddings[source]
    gold_delta = embeddings[gold] - embeddings[source]
    pred_norm = float(np.linalg.norm(pred_delta))
    gold_norm = float(np.linalg.norm(gold_delta))
    if pred_norm <= 1e-9 or gold_norm <= 1e-9:
        return 0.0, 0.0, 0.0
    direction = float(
        np.clip(np.dot(pred_delta, gold_delta) / (pred_norm * gold_norm), 0.0, 1.0)
    )
    precision = direction * min(1.0, gold_norm / pred_norm)
    recall = direction * min(1.0, pred_norm / gold_norm)
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1