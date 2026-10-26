import pytest

from weva import build_agent, create_env, load_config, run_benchmark, run_episode, summarise
from weva.evaluation import detection_lead, ler, macro_ler, steps_to_threshold, transition_fidelity


def run(method, **env_kwargs):
    cfg = load_config([f"configs/methods/{method}.yaml", "configs/toy.yaml"])
    return run_benchmark(
        build_agent(cfg), create_env("toy-ledger", **env_kwargs), max_steps=cfg.run.max_steps
    )


def test_react_edits_the_old_ledger_and_violates():
    rs = run("react")
    assert all(r.violation and not r.success for r in rs)


def test_weva_predicts_checks_then_acts():
    rs = run("weva")
    assert all(r.success and not r.violation for r in rs)
    assert all(r.env_steps == 3 for r in rs)
    first = rs[0].records[0]
    assert first.meta_action == "LOOKAHEAD" and first.n_pruned >= 1 and first.lookahead_changed_action
    assert rs[0].internal_calls > rs[0].env_steps


def test_h0_monitors_but_does_not_search():
    rs = run("weva_h0")
    assert all(r.records[0].n_expanded == 0 for r in rs)
    assert not all(r.success for r in rs)
    assert any(t.predicted is not None for r in rs for t in r.transitions)


def test_free_text_state_cannot_verify_preconditions():
    s = summarise(run("free_text_pred"))
    assert s["success"] == 0.0


def test_model_trust_drops_under_dynamics_shift():
    cfg = load_config(["configs/methods/weva.yaml", "configs/toy.yaml"])
    agent = build_agent(cfg)
    env = create_env("toy-ledger", shift_after=1)
    r = run_episode(agent, env, env.tasks()[0], max_steps=6)
    assert not r.success
    assert r.transitions[0].agreement < 0.7
    assert r.records[1].trust < 0.7 and r.records[-1].trust < 0.7


def test_metrics():
    rs = run("weva")
    assert ler(rs) == 0.0 and macro_ler({"toy": rs}) == 0.0
    tf = transition_fidelity([t for r in rs for t in r.transitions])
    assert tf["overall"] == pytest.approx(1.0)
    assert detection_lead([1, 1, 0.4, 0.3], [1, 1, 1, 0.4], alert=0.5, degradation=0.5) == 1
    assert steps_to_threshold([0.1, 0.5, 0.9], 0.5) == 2


def test_every_executed_action_is_recorded_and_trust_resets():
    cfg = load_config(["configs/methods/weva.yaml", "configs/toy.yaml"])
    agent = build_agent(cfg)
    env = create_env("toy-ledger")
    res = run_episode(agent, env, env.tasks()[0], cfg.run.max_steps)
    assert len(res.transitions) == res.env_steps == len(res.records)
    assert [t.action for t in res.transitions] == [r.action for r in res.records]
    last = res.transitions[-1]
    assert last.next_observation and last.target_prompt and last.actor_prompt and last.success is not None
    assert agent.trust.n > 0
    agent.start_episode("next")
    assert agent.trust.n == 0 and agent.trust.mu == cfg.controller.initial_trust


def test_group_rollouts_start_from_the_same_agent_state():
    from weva.training import collect_round

    cfg = load_config(["configs/methods/weva.yaml", "configs/toy.yaml"])
    agent = build_agent(cfg)
    env = create_env("toy-ledger")
    starts = []
    original = agent.start_episode

    def spy(goal):
        starts.append((len(agent.memory.episodic), dict(agent.memory.state_dict()["semantic"])))
        original(goal)

    agent.start_episode = spy
    _, _, results = collect_round(agent, env, env.tasks()[:1], 3, cfg.run.max_steps)
    assert len(results) == 3 and all(s == starts[0] for s in starts)
    assert [r.records[0].action for r in results] == [results[0].records[0].action] * 3
    assert len(agent.memory.episodic) == sum(len(r.transitions) for r in results)


def test_internal_calls_are_counted_per_episode():
    cfg = load_config(["configs/methods/weva.yaml", "configs/toy.yaml"])
    agent = build_agent(cfg)
    env = create_env("toy-ledger")
    task = env.tasks()[0]
    agent.freeze()
    calls = [run_episode(agent, env, task, cfg.run.max_steps).internal_calls for _ in range(3)]
    assert calls[0] > 0 and calls == [calls[0]] * 3


def test_transitions_keep_the_actor_proposal_that_produced_the_executed_action():
    from weva.components import parse_candidates

    cfg = load_config(["configs/methods/weva.yaml", "configs/toy.yaml"])
    env = create_env("toy-ledger")
    res = run_episode(build_agent(cfg), env, env.tasks()[0], cfg.run.max_steps)
    first = res.transitions[0]
    cands = parse_candidates(first.actor_output, cfg.lookahead.branching)
    assert first.action in cands and cands[0] != first.action and res.records[0].lookahead_changed_action
    assert all(t.actor_prompt and t.action in t.actor_output for t in res.transitions)


def test_interaction_window_carries_actions_and_their_observations():
    from weva.agent import WEvaAgent
    from weva.envs.toy_llm import make_toy_llm

    llm = make_toy_llm()
    agent = WEvaAgent(load_config(["configs/methods/weva.yaml", "configs/toy.yaml"]), llm)
    env = create_env("toy-ledger")
    run_episode(agent, env, env.tasks()[0], max_steps=3)
    prompts = [p for role, p, _ in llm.log if role == "perceptor"]
    window = prompts[-1].split("RECENT INTERACTION:", 1)[1].split("\n", 1)[0]
    assert "0: edit ledger row 4 = Short delivery -> " in window and "LEDGER_ROW4: Short delivery" in window
