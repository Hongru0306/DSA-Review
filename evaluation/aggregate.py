"""Metric aggregation: macro-mean x100, Ret.Avg / Gen.Avg composites.

Conventions matching the experiment tables:
- retrieval: ``Ret.Avg = (ACC + MRR + nDCG) / 3`` (or a 4-way mean including Coverage).
- generation: ``Gen.Avg = mean(char_f1, token_f1, ...)``, the mean over components.
"""

from __future__ import annotations

from collections.abc import Iterable
from statistics import mean
from typing import Any


def macro_mean(values: Iterable[float]) -> float:
    """Macro-mean (per-query metrics averaged)."""
    return mean(values) if values else 0.0


def mean_100(values: Iterable[float]) -> float:
    """Macro-mean multiplied by 100 for display."""
    return mean(values) * 100 if values else 0.0


def ret_avg(acc: float, mrr: float, ndcg: float, coverage: float | None = None) -> float:
    """Retrieval composite average."""

    if coverage is None:
        return (acc + mrr + ndcg) / 3.0
    return (coverage + mrr + ndcg) / 3.0


def gen_avg(components: Iterable[float]) -> float:
    """Generation composite average (mean over the given score components)."""
    return mean(components) if components else 0.0


def summarize_metric_rows(
    rows: list[dict[str, Any]],
    key_aliases: dict[str, str] | None = None,
) -> dict[str, float]:
    """Macro-average numeric keys over a list of row dicts (returned in %).

    ``key_aliases`` renames keys, e.g. ``{"acc": "ACC", "mrr": "MRR"}``.
    """
    if not rows:
        return {}
    keys: list[str] = []
    for row in rows:
        for key, value in row.items():
            if isinstance(value, (int, float)) and key not in keys:
                keys.append(key)
    aliases = key_aliases or {}
    out: dict[str, float] = {}
    for key in keys:
        values = [float(row[key]) for row in rows if key in row]
        out[aliases.get(key, key)] = mean_100(values)
    return out


def summarize_by_group(
    rows: list[dict[str, Any]],
    group_key: str,
    aliases: dict[str, str] | None = None,
) -> dict[Any, dict[str, float]]:
    """Group rows by a field (e.g. hop) and macro-average each group."""
    groups: dict[Any, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(row.get(group_key), []).append(row)
    return {group: summarize_metric_rows(member_rows, aliases) for group, member_rows in groups.items()}