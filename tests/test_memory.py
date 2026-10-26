from weva.config import MemoryConfig
from weva.memory import MemoryManager
from weva.types import State, Transition

RULE = "ledger_2025 must not change"


def checker(c, source, action, target):
    return c == RULE and "ledger_2025" in action


def tr(action, failure, step=0):
    s = State(V=("files",), C=(RULE,))
    return Transition(
        source=s,
        action=action,
        target=State(V=("files", "x")),
        step=step,
        failure=failure,
        predicted=State(V=("files",)),
    )


def proposer(t, similar):
    return RULE


def test_candidate_needs_recurrence_and_extends_without_regressing():
    m = MemoryManager(MemoryConfig(recurrence_min=2, promote_threshold=2), checker=checker, proposer=proposer)
    m.record(tr("edit ledger_2025 row 4", failure=True))
    assert not m.preconditions
    m.record(tr("edit ledger_2025 row 4", failure=True))
    assert RULE in m.preconditions and not m.preconditions[RULE].active
    m.record(tr("edit ledger row 4", failure=False))
    out = m.end_episode()
    assert out["promoted"] == [RULE] and m.active_preconditions() == [RULE]


def test_candidate_that_would_reject_an_earlier_success_is_not_promoted():
    m = MemoryManager(MemoryConfig(recurrence_min=1, promote_threshold=2), checker=checker, proposer=proposer)
    for _ in range(3):
        m.record(tr("edit ledger_2025 row 4", failure=True))
    m.record(tr("edit ledger_2025 row 9", failure=False))
    out = m.end_episode()
    assert not out["promoted"] and RULE not in m.preconditions


def test_contradicted_candidate_is_dropped():
    m = MemoryManager(
        MemoryConfig(recurrence_min=1, promote_threshold=2, demote_threshold=-1),
        checker=checker,
        proposer=proposer,
    )
    m.record(tr("edit ledger_2025 row 4", failure=True))
    m.record(tr("edit ledger_2025 row 9", failure=False))
    m.record(tr("edit ledger_2025 row 8", failure=False))
    m.end_episode()
    assert RULE not in m.preconditions


def test_frozen_memory_does_not_write_and_levels_are_respected():
    m = MemoryManager(MemoryConfig(levels=("episodic",)), checker=checker, proposer=proposer)
    m.freeze()
    m.record(tr("a", False))
    assert not m.episodic
    m.freeze(False)
    m.record(tr("a", True))
    m.record(tr("a", True))
    assert m.episodic and not m.preconditions and not m.active_preconditions()


def test_retrieval_policy_depends_on_trust_and_state_roundtrip(tmp_path):
    m = MemoryManager(MemoryConfig())
    m.record(tr("edit ledger row 4", False))
    assert m.retrieve(State(V=("files",)), trust=0.9).text == ""
    assert "edit ledger row 4" in m.retrieve(State(V=("files",)), trust=0.5).text
    m.save(tmp_path / "m.json")
    m2 = MemoryManager(MemoryConfig())
    m2.load(tmp_path / "m.json")
    assert dict(m2.semantic) == dict(m.semantic)


BROAD = "ledger_2025 must not change"
NARROW = "row 4 of ledger_2025 must not change"


def revision_checker(c, source, action, target):
    if c == BROAD:
        return "ledger_2025" in action
    if c == NARROW:
        return "ledger_2025 row 4" in action
    return False


def test_revised_precondition_respects_successes_older_than_any_window():
    proposals = iter([BROAD, NARROW, NARROW, NARROW])
    m = MemoryManager(
        MemoryConfig(recurrence_min=1, promote_threshold=1),
        checker=revision_checker,
        proposer=lambda t, similar: next(proposals),
    )
    m.record(tr("edit ledger_2025 row 9", failure=False))
    for i in range(120):
        m.record(tr(f"read file {i}", failure=False))
    m.record(tr("edit ledger_2025 row 4", failure=True))
    out = m.end_episode()
    assert BROAD in out["removed"] and BROAD not in m.preconditions
    m.record(tr("edit ledger_2025 row 4", failure=True))
    out = m.end_episode()
    assert out["promoted"] == [NARROW] and m.active_preconditions() == [NARROW]
    for _ in range(4):
        m.record(tr("edit ledger_2025 row 4 again", failure=False))
    out = m.end_episode()
    assert NARROW in out["removed"] and not m.active_preconditions()


def test_candidate_that_rejects_no_observed_failure_is_not_promoted():
    m = MemoryManager(
        MemoryConfig(recurrence_min=1, promote_threshold=1),
        checker=lambda c, s, a, t: False,
        proposer=lambda t, similar: "an unrelated rule",
    )
    m.record(tr("edit ledger_2025 row 4", failure=True))
    m.record(tr("edit ledger_2025 row 4", failure=True))
    assert m.preconditions["an unrelated rule"].evidence >= 1
    out = m.end_episode()
    assert not out["promoted"] and not m.active_preconditions()
