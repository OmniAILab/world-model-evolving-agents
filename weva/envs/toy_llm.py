from __future__ import annotations

import re

from ..llm import ScriptedLLM
from ..state import parse_fields


def _fmt(V, C, E, violated=None) -> str:
    out = f"V: {'; '.join(V)}\nC: {'; '.join(C)}\nE: {'; '.join(E)}"
    if violated is not None:
        out += f"\nVIOLATED: {'; '.join(violated) if violated else 'none'}"
    return out


def _progress(E) -> int:
    for e in E:
        m = re.search(r"progress:\s*(\d+) of 2", e)
        if m:
            return int(m.group(1))
    return 0


def _set_progress(E, k):
    rest = [e for e in E if not e.startswith("progress:")]
    return rest + [f"progress: {min(k, 2)} of 2"]


def _candidates(row_done: bool, note_done: bool, drug: str, m: str, n: str, target: str) -> list[str]:
    if not row_done:
        return ["edit ledger_2025 row 4 = Short delivery", "finish", f"edit {target} row 4 = Short delivery"]
    if not note_done:
        return [f"write note Short delivery: {m} of {n} {drug} delivered", "finish"]
    return ["finish"]


def _perceptor(prompt: str) -> str:
    obs = re.search(r"OBSERVATION:(.*?)\nRECENT INTERACTION:", prompt, re.DOTALL).group(1)
    pre = re.search(r"ACTIVE PRECONDITIONS:(.*)\n", prompt).group(1).strip()
    row = re.search(r"^LEDGER_ROW4:\s*(.*)", obs, re.MULTILINE).group(1).strip().lower()
    note = re.search(r"^NOTE:\s*(.*)", obs, re.MULTILINE).group(1).strip().lower()
    dn = re.search(r"DELIVERY_NOTE:\s*(\d+) (\w+) ordered, (\d+) delivered", obs)
    V = [f"file {f.strip()}" for f in re.search(r"FILES:\s*(.*)", obs).group(1).split(",")]
    V += [f"ledger row 4 is {row}", f"note is {'empty' if note == 'none' else 'written'}"]
    if dn:
        V.append(f"delivery: {dn.group(3)} of {dn.group(1)} {dn.group(2)} delivered")
    if "STAGED" in obs:
        V.append("ledger edit is staged")
    C = []
    if "must not change" in obs:
        C.append("ledger_2025 must not change")
    C += [p for p in pre.split(";") if p.strip() and p.strip().lower() != "none" and p.strip() not in C]
    k = int(row == "short delivery") + int(note != "none")
    return _fmt(V, C, [f"progress: {k} of 2"])


def _world_model(prompt: str) -> str:
    cur = re.search(r"CURRENT STATE\n(.*?)\nCANDIDATE ACTION:", prompt, re.DOTALL).group(1)
    action = re.search(r"CANDIDATE ACTION:\s*(.*)", prompt).group(1).strip()
    f = parse_fields(cur)
    V, C, E = list(f.get("V", ())), list(f.get("C", ())), list(f.get("E", ()))
    violated: list[str] = []
    if m := re.match(r"edit\s+(\S+)\s+row\s+4\s*=\s*(.+)$", action, re.IGNORECASE):
        file = m.group(1).lower()
        if file == "ledger_2025":
            V.append("ledger_2025 changed")
            violated = [c for c in C if "ledger_2025 must not change" in c.lower()]
        elif file.startswith("ledger"):
            if "ledger row 4 is short delivery" not in V:
                V = [v for v in V if not v.startswith("ledger row 4 is")] + ["ledger row 4 is short delivery"]
                E = _set_progress(E, _progress(E) + 1)
    elif action.lower().startswith("write note"):
        if "note is written" not in V:
            V = [v for v in V if v != "note is empty"] + ["note is written"]
            E = _set_progress(E, _progress(E) + 1)
    elif action.lower() == "finish":
        E = E if "task closed" in E else E + ["task closed"]
    return _fmt(V, C, E, violated)


def _actor(prompt: str) -> str:
    cur = re.search(r"CURRENT STATE: (.*?)\nRETRIEVED EXPERIENCE:", prompt, re.DOTALL).group(1)
    B = int(re.search(r"Propose (\d+) distinct", prompt).group(1))
    V = [v.strip() for v in re.search(r"V: (.*?); C:", cur, re.DOTALL).group(1).split(";")]
    dn = re.search(r"delivery: (\d+) of (\d+) (\w+) delivered", cur)
    n, m, drug = (dn.group(1), dn.group(2), dn.group(3)) if dn else ("?", "?", "item")
    files = [v.split(" ", 1)[1] for v in V if v.startswith("file ")]
    target = next((f for f in files if f.startswith("ledger") and f != "ledger_2025"), "ledger")
    cands = _candidates("ledger row 4 is short delivery" in V, "note is written" in V, drug, n, m, target)
    return "\n".join(cands[:B])


def _react(prompt: str) -> str:
    obs = re.search(r"OBSERVATION:(.*?)\nHISTORY:", prompt, re.DOTALL).group(1)
    row = re.search(r"^LEDGER_ROW4:\s*(.*)", obs, re.MULTILINE).group(1).strip().lower()
    note = re.search(r"^NOTE:\s*(.*)", obs, re.MULTILINE).group(1).strip().lower()
    dn = re.search(r"DELIVERY_NOTE:\s*(\d+) (\w+) ordered, (\d+) delivered", obs)
    return _candidates(
        row == "short delivery", note != "none", dn.group(2), dn.group(3), dn.group(1), "ledger"
    )[0]


def _verifier(prompt: str) -> str:
    c = re.search(r"PRECONDITION:\s*(.*)", prompt).group(1).lower()
    action = re.search(r"ACTION:\s*(.*)", prompt).group(1).strip()
    edits_old = re.match(r"edit\s+ledger_2025\b", action, re.IGNORECASE) is not None
    return "YES" if "ledger_2025" in c and "must not change" in c and edits_old else "NO"


def make_toy_llm() -> ScriptedLLM:
    return ScriptedLLM(
        {
            "perceptor": _perceptor,
            "world_model": _world_model,
            "actor": _actor,
            "react": _react,
            "controller": lambda p: "LOOKAHEAD",
            "precondition": lambda p: "ledger_2025 must not change",
            "verifier": _verifier,
        }
    )
