from __future__ import annotations

import re
from collections.abc import Iterable, Mapping

from .types import Prediction, State

_NULL = {"", "none", "n/a", "-", "null"}
_FIELD = re.compile(r"^\s*(VIOLATED|V|C|E)\s*:\s*(.*)$", re.IGNORECASE)


def split_entries(text: str) -> tuple[str, ...]:
    out: list[str] = []
    for part in re.split(r"[;\n]", text):
        p = part.strip().lstrip("-*• ").strip()
        if p.lower() not in _NULL:
            out.append(p)
    return tuple(out)


def parse_fields(text: str) -> dict[str, tuple[str, ...]]:
    fields: dict[str, tuple[str, ...]] = {}
    cur: str | None = None
    for line in text.splitlines():
        m = _FIELD.match(line)
        if m:
            cur = m.group(1).upper()
            fields[cur] = fields.get(cur, ()) + split_entries(m.group(2))
        elif cur is not None and line.strip():
            fields[cur] = fields[cur] + split_entries(line)
    return fields


def parse_state(text: str, mode: str = "quasi_symbolic") -> State:
    if mode == "free_text":
        return State(V=(" ".join(text.split()),))
    f = parse_fields(text)
    return State(V=f.get("V", ()), C=f.get("C", ()), E=f.get("E", ()))


def parse_prediction(text: str, source: State | None = None, mode: str = "quasi_symbolic") -> Prediction:
    if mode == "free_text":
        return Prediction(State(V=(" ".join(text.split()),)), ())
    f = parse_fields(text)
    if not any(k in f for k in ("V", "C", "E")):
        return Prediction(source if source is not None else State(), ())
    base = source if source is not None else State()
    state = State(
        V=f.get("V", base.V),
        C=f.get("C", base.C),
        E=f.get("E", base.E),
    )
    return Prediction(state, f.get("VIOLATED", ()))


_DETERMINERS = {"a", "an", "the"}
_SUFFIXES = ("ing", "ed", "es", "s")
TERMINAL_MARKERS = ("task closed", "terminal", "episode over", "finished", "task complete")


def _stem(word: str) -> str:
    for suf in _SUFFIXES:
        if word.endswith(suf) and len(word) - len(suf) >= 3:
            word = word[: -len(suf)]
            break
    if word.endswith("e") and len(word) > 3:
        word = word[:-1]
    return word


def plain(entry: str) -> str:
    text = re.sub(r"[^\w\s]", " ", entry.lower())
    tokens = [_stem(t) for t in text.split() if t not in _DETERMINERS]
    return " ".join(tokens)


def plain_set(entries: Iterable[str]) -> frozenset[str]:
    return frozenset(p for p in (plain(e) for e in entries) if p)


def jaccard(a: frozenset[str] | set[str], b: frozenset[str] | set[str]) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def field_similarity(x: Iterable[str], y: Iterable[str]) -> float:
    return jaccard(plain_set(x), plain_set(y))


def strip_terminal(entries: Iterable[str], markers: tuple[str, ...] = TERMINAL_MARKERS) -> tuple[str, ...]:
    return tuple(e for e in entries if not any(m in e.lower() for m in markers))


def sim_psi(
    s1: State,
    s2: State,
    fields: tuple[str, ...] = ("V", "C", "E"),
    weights: Mapping[str, float] | None = None,
    exclude_terminal: bool = True,
) -> float:
    weights = weights or {}
    num = den = 0.0
    for f in fields:
        a, b = s1.get(f), s2.get(f)
        if f == "E" and exclude_terminal:
            a, b = strip_terminal(a), strip_terminal(b)
        w = float(weights.get(f, 1.0))
        num += w * field_similarity(a, b)
        den += w
    return num / den if den else 1.0


_RATIO = re.compile(r"(\d+(?:\.\d+)?)\s*(?:of|/)\s*(\d+(?:\.\d+)?)", re.IGNORECASE)
_FLOAT = re.compile(r"progress\s*[:=]?\s*(\d*\.\d+|\d+)", re.IGNORECASE)


def default_progress(state: State) -> float:
    best = 0.0
    for entry in state.E:
        m = _RATIO.search(entry)
        if m and float(m.group(2)) > 0:
            best = max(best, min(1.0, float(m.group(1)) / float(m.group(2))))
            continue
        m = _FLOAT.search(entry)
        if m:
            best = max(best, min(1.0, float(m.group(1))))
    return best
