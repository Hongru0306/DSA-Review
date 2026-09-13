"""指标聚合:宏平均 ×100、Ret.Avg / Gen.Avg 复合。

口径与实验表格一致:
- 检索:``Ret.Avg = (ACC + MRR + nDCG) / 3``(或含 Coverage 的四元均值)。
- 生成:``Gen.Avg = mean(char_f1, token_f1, ...)`` 的三 / 多件套均值。
"""

from __future__ import annotations

from collections.abc import Iterable
from statistics import mean
from typing import Any


def macro_mean(values: Iterable[float]) -> float:
    """宏平均(每题指标后取均值)。"""
    return mean(values) if values else 0.0


def mean_100(values: Iterable[float]) -> float:
    """宏平均后乘 100 展示。"""
    return mean(values) * 100 if values else 0.0


def ret_avg(acc: float, mrr: float, ndcg: float, coverage: float | None = None) -> float:
    """检索复合平均。"""

    if coverage is None:
        return (acc + mrr + ndcg) / 3.0
    return (coverage + mrr + ndcg) / 3.0


def gen_avg(components: Iterable[float]) -> float:
    """生成复合平均(对若干分数分量取均值)。"""
    return mean(components) if components else 0.0


def summarize_metric_rows(
    rows: list[dict[str, Any]],
    key_aliases: dict[str, str] | None = None,
) -> dict[str, float]:
    """对每行词典在数值键上求宏平均(% 展示)。

    ``key_aliases`` 统一别名,例如 ``{"acc": "ACC", "mrr": "MRR"}``。
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
    """按某个字段(如 hop)分组后逐组宏平均。"""
    groups: dict[Any, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(row.get(group_key), []).append(row)
    return {group: summarize_metric_rows(member_rows, aliases) for group, member_rows in groups.items()}