import pytest

pytest.importorskip("scienceworld")

from weva import load_config, run_episode
from weva.agent import WEvaAgent
from weva.envs import create_env
from weva.llm import ScriptedLLM


@pytest.fixture(scope="module")
def env():
    e = create_env("scienceworld", task_names=["find-living-thing"], max_variations=1)
    yield e
    e.close()


def test_gold_path_succeeds_without_violations_and_invalid_actions_are_verified(env):
    task = env.tasks("test")[0]
    env.reset(task)
    env.sw.load(task.metadata["task"], task.metadata["variation"], env.simplification, generateGoldPath=True)
    gold = env.sw.get_gold_action_sequence()
    env.reset(task)
    _, _, info = env.step("fly to the moon")
    assert info["violation"]
    env.reset(task)
    for a in gold:
        _, done, info = env.step(a)
        assert not info["violation"]
        if done:
            break
    out = env.evaluate()
    assert out["success"] and not out["violation"] and out["score"] == 1.0


def test_weva_loop_runs_on_scienceworld(env):
    llm = ScriptedLLM(
        {
            "perceptor": lambda p: "V: agent in a room\nC: none\nE: progress: 0 of 1",
            "world_model": lambda p: "V: agent in a room\nC: none\nE: progress: 0 of 1\nVIOLATED: none",
            "actor": lambda p: "look around\nfly to the moon\nwait",
            "controller": lambda p: "LOOKAHEAD",
            "precondition": lambda p: "PRECONDITION: none",
            "verifier": lambda p: "NO",
        }
    )
    cfg = load_config(["configs/methods/weva.yaml", "configs/toy.yaml"])
    res = run_episode(WEvaAgent(cfg, llm), env, env.tasks("test")[0], max_steps=3)
    assert res.env_steps == 3 and len(res.records) == 3 and "score" in res.extra


def test_only_a_terminal_verified_violation_counts_for_ler(env):
    task = env.tasks("test")[0]
    env.reset(task)
    _, done, info = env.step("fly to the moon")
    assert info["violation"] and info["invalid_action"] and not done
    assert env.evaluate()["violation"] is False and env.evaluate()["invalid_actions"] == 1
    _, done, info = env.step("focus on air")
    assert done and info["task_failed"]
    out = env.evaluate()
    assert out["violation"] is True and out["success"] is False
