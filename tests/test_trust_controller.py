import numpy as np

from weva.components import LearnedController, ThresholdController
from weva.training import Rollout, controller_reward, reinforce_update
from weva.trust import ModelTrust
from weva.types import ControlState, MetaAction, State


def test_trust_window_and_field_diagnosis():
    tr = ModelTrust(window=2, initial=0.7)
    assert tr.mu == 0.7 and tr.diagnose() is None
    pred = State(V=("a",), C=("c",), E=("progress: 1 of 2",))
    tr.update(pred, pred)
    tr.update(pred, State(V=("a",), C=("other",), E=("progress: 1 of 2",)))
    assert tr.diagnose() == "C" and tr.diagnose(field_level=False) == "scalar"
    assert 0.0 < tr.mu < 1.0
    tr.update(pred, pred)
    tr.update(pred, pred)
    assert tr.mu == 1.0


def test_threshold_controller_follows_trust_bands():
    c = ThresholdController()
    assert c.select(ControlState(trust=0.9)) == MetaAction.LOOKAHEAD
    assert c.select(ControlState(trust=0.45, novelty=0.9)) == MetaAction.RETRIEVE
    assert c.select(ControlState(trust=0.2)) == MetaAction.ABSTRACT
    assert c.select(ControlState(trust=0.2)) == MetaAction.ROLLBACK


def test_reinforce_increases_probability_of_rewarded_action():
    ctrl = LearnedController(seed=0)
    x = [0.5, 0.0, 0.5, 0.0, 0.5]
    a_good = ctrl.ACTIONS.index(MetaAction.ACT)
    before = ctrl.probs(np.array(x))[a_good]
    for _ in range(50):
        good = Rollout(features=[x], action_idx=[a_good], performance=1.0)
        bad = Rollout(features=[x], action_idx=[0], performance=0.0)
        reinforce_update(ctrl, [good, bad], lr=0.5)
    assert ctrl.probs(np.array(x))[a_good] > before


def test_controller_reward_penalises_unchanged_lookahead():
    r = Rollout(performance=1.0, lookahead_total=4, lookahead_unchanged=4)
    assert controller_reward(r, lam=0.2) == 0.8


def test_restoring_a_snapshot_keeps_controller_exploration():
    from weva.agent import WEvaAgent
    from weva.config import load_config
    from weva.envs.toy_llm import make_toy_llm
    from weva.training import _RecordingController

    cfg = load_config(["configs/methods/weva.yaml", "configs/toy.yaml"])
    agent = WEvaAgent(cfg, make_toy_llm(), controller=_RecordingController(LearnedController(seed=0)))
    snap = agent.snapshot()
    seqs = []
    for _ in range(4):
        agent.restore(snap)
        seqs.append(tuple(agent.controller.select(ControlState()) for _ in range(8)))
    assert len(set(seqs)) > 1
