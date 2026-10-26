from weva.components import Actor, WorldModel
from weva.config import LookaheadConfig
from weva.llm import ScriptedLLM
from weva.search import VerifiedLookahead, count_violations
from weva.types import Prediction, State


def make_llm():
    def actor(prompt):
        return "shortcut\nsafe step\nidle"

    def wm(prompt):
        action = prompt.split("CANDIDATE ACTION:")[1].split("\n")[0].strip()
        if action == "shortcut":
            return "V: s\nC: do not touch X\nE: progress: 1 of 1\nVIOLATED: do not touch X"
        if action == "safe step":
            return "V: s\nC: do not touch X\nE: progress: 1 of 2\nVIOLATED: none"
        return "V: s\nC: do not touch X\nE: progress: 0 of 2\nVIOLATED: none"

    return ScriptedLLM({"actor": actor, "world_model": wm})


def search(verify, horizon=1):
    llm = make_llm()
    cfg = LookaheadConfig(horizon=horizon, branching=3, beam=2, verify=verify)
    return VerifiedLookahead(Actor(llm), WorldModel(llm), cfg)


S0 = State(V=("s",), C=("do not touch X",), E=("progress: 0 of 2",))


def test_verification_prunes_violating_branch():
    r = search(verify=True).search("g", S0)
    assert r.action == "safe step" and r.n_pruned == 1 and r.default_action == "shortcut"


def test_without_verification_takes_the_violating_branch():
    r = search(verify=False).search("g", S0)
    assert r.action == "shortcut"


def test_no_survivor_returns_none_so_caller_falls_back():
    llm = ScriptedLLM(
        {
            "actor": lambda p: "a\nb",
            "world_model": lambda p: "V: s\nC: do not touch X\nE:\nVIOLATED: do not touch X",
        }
    )
    la = VerifiedLookahead(Actor(llm), WorldModel(llm), LookaheadConfig(horizon=2, branching=2, beam=2))
    assert la.search("g", S0).action is None


def test_later_violation_prunes_first_step_branch():
    def wm(prompt):
        if "V: b" in prompt.split("CANDIDATE ACTION:")[0]:
            return "V: z\nC: do not touch X\nE: progress: 1 of 1\nVIOLATED: do not touch X"
        if "CANDIDATE ACTION: first" in prompt:
            return "V: b\nC: do not touch X\nE: progress: 1 of 2\nVIOLATED: none"
        return "V: c\nC: do not touch X\nE: progress: 0 of 2\nVIOLATED: none"

    llm = ScriptedLLM({"actor": lambda p: "first\nsecond", "world_model": wm})
    la = VerifiedLookahead(Actor(llm), WorldModel(llm), LookaheadConfig(horizon=2, branching=2, beam=2))
    r = la.search("g", S0)
    assert r.action == "second"


def test_count_violations_matches_active_preconditions_only():
    src = State(C=("ledger_2025 must not change",))
    assert count_violations(src, Prediction(State(), ("Ledger_2025 must not change.",))) == 1
    assert count_violations(src, Prediction(State(), ("something unrelated",))) == 0


RULE = "do not touch X"


def make_wm(keep_rule: bool):
    c = RULE if keep_rule else "none"

    def wm(prompt):
        head, tail = prompt.split("CANDIDATE ACTION:")
        action = tail.split("\n")[0].strip()
        if "V: b" in head:
            return f"V: z\nC: {c}\nE: progress: 2 of 2\nVIOLATED: none"
        if action == "first":
            return f"V: b\nC: {c}\nE: progress: 1 of 2\nVIOLATED: none"
        return f"V: c\nC: {c}\nE: progress: 1 of 2\nVIOLATED: none"

    return wm


def touches_x_after_b(c, source, action, target):
    return c == RULE and "b" in source.V


def two_step(verifier="both", checker=touches_x_after_b, keep_rule=True):
    llm = ScriptedLLM({"actor": lambda p: "first\nsecond", "world_model": make_wm(keep_rule)})
    cfg = LookaheadConfig(horizon=2, branching=2, beam=2, verifier=verifier)
    return VerifiedLookahead(Actor(llm), WorldModel(llm), cfg, checker=checker)


def test_checker_verifies_against_preconditions_without_a_declared_violation():
    r = two_step().search("g", S0)
    assert r.action == "second" and r.n_pruned == 2


def test_declared_only_mode_misses_what_the_checker_catches():
    r = two_step(verifier="declared").search("g", S0)
    assert r.action == "first" and r.n_pruned == 0


def test_a_precondition_the_action_changes_no_longer_applies_downstream():
    r = two_step(keep_rule=False).search("g", S0)
    assert r.action == "first" and r.n_pruned == 0


def test_promoted_preconditions_stay_active_when_the_prediction_drops_them():
    r = two_step(keep_rule=False).search("g", S0, required=[RULE])
    assert r.action == "second" and r.n_pruned == 2


def test_count_violations_combines_declared_and_checked():
    src = State(V=("b",), C=(RULE, "keep the door shut"))
    pred = Prediction(State(), ("keep the door shut",))
    assert count_violations(src, pred, action="go", checker=touches_x_after_b, mode="both") == 2
    assert count_violations(src, pred, action="go", checker=touches_x_after_b, mode="checker") == 1
    assert count_violations(src, pred, action="go", checker=touches_x_after_b, mode="declared") == 1
