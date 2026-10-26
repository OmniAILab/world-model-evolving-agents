from weva.state import default_progress, parse_prediction, parse_state, plain, sim_psi, strip_terminal
from weva.types import State


def test_parse_state_roundtrip():
    s = parse_state("V: a; b\nC: must not change x\nE: progress: 1 of 2")
    assert s.V == ("a", "b") and s.C == ("must not change x",) and s.E == ("progress: 1 of 2",)
    assert parse_state(s.render()) == s


def test_parse_prediction_carries_preconditions_and_violations():
    src = State(V=("x",), C=("c1",), E=())
    p = parse_prediction("V: y\nE: progress: 1 of 2\nVIOLATED: c1", source=src)
    assert p.state.C == ("c1",) and p.violated == ("c1",)
    assert parse_prediction("garbage", source=src).state == src


def test_plain_normalises_surface_form():
    assert plain("The ledger changed.") == plain("ledger changes")
    assert sim_psi(State(V=("The Files",)), State(V=("file",))) == 1.0


def test_sim_psi_excludes_terminal_outcome():
    a = State(E=("progress: 1 of 2", "task closed"))
    b = State(E=("progress: 1 of 2",))
    assert sim_psi(a, b) == 1.0
    assert strip_terminal(a.E) == ("progress: 1 of 2",)


def test_progress_readout():
    assert default_progress(State(E=("progress: 1 of 2",))) == 0.5
    assert default_progress(State(E=("progress: 0.25",))) == 0.25
    assert default_progress(State()) == 0.0
