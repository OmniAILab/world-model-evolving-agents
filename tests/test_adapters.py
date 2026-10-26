import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from weva.envs import create_env
from weva.envs.adapters import AppWorldEnv, WebShopEnv
from weva.evaluation import run_episode, scenario_goal_completion, summarise
from weva.types import EpisodeResult


class FakeWebShop:
    def __init__(self):
        self.server = SimpleNamespace(goals=[{}] * 3, user_sessions={})
        self.instruction_text = "buy a red mug under 10 dollars"

    def reset(self, session):
        self.session = str(session)
        self.page = "search"
        self.server.user_sessions[self.session] = {}
        return "WebShop [SEP] Search", None

    def get_available_actions(self):
        if self.page == "search":
            return {"has_search_bar": True, "clickables": []}
        return {"has_search_bar": False, "clickables": ["b01", "buy now"]}

    def step(self, action):
        if action.startswith("search["):
            self.page = "results"
            return "results [SEP] b01 red mug $12", 0.0, False, None
        if action == "click[buy now]":
            self.server.user_sessions[self.session]["verbose_info"] = {"r_price": False}
            return "Thank you", 0.6, True, None
        return "results", 0.0, False, None


def test_webshop_score_invalid_actions_and_budget_violation():
    env = WebShopEnv(env=FakeWebShop())
    task = env.tasks("test")[0]
    obs = env.reset(task)
    assert task.goal.startswith("buy a red mug") and "search[<query>]" in obs
    _, _, info = env.step("click[nothing]")
    assert info["violation"]
    env.step("search[red mug]")
    _, done, info = env.step("click[buy now]")
    out = env.evaluate()
    assert done and not info["violation"] and out["score"] == 0.6 and not out["success"] and out["violation"]


class FakeWorld:
    def __init__(self, task_id, collateral):
        self.task = SimpleNamespace(
            instruction="pay the rent", supervisor=SimpleNamespace(first_name="Ann", last_name="Lee")
        )
        self.collateral = collateral
        self.done = False

    def execute(self, code):
        if "bad" in code:
            return "Execution failed. Traceback:\nException: Response status code is 422"
        if "complete_task" in code:
            self.done = True
        return "Execution successful."

    def task_completed(self):
        return self.done

    def evaluate(self):
        failures = [{"requirement": "no other changes", "label": "no_op_pass"}] if self.collateral else []
        return SimpleNamespace(success=not self.collateral, passes=[{}], failures=failures)

    def close(self):
        pass


def test_appworld_api_errors_tgc_sgc_and_collateral_violation():
    ids = {"test_normal": ["s1_1", "s1_2", "s2_1"]}
    worlds = {"s1_1": False, "s1_2": True, "s2_1": False}
    env = AppWorldEnv(world_factory=lambda i: FakeWorld(i, worlds[i]), task_ids=ids)
    results = []
    for task in env.tasks("test"):
        obs = env.reset(task)
        assert "pay the rent" in obs and task.goal == "pay the rent"
        _, _, info = env.step("```python\napis.bad()\n```")
        assert info["violation"]
        _, done, _ = env.step("apis.supervisor.complete_task()")
        assert done
        out = env.evaluate()
        results.append(EpisodeResult(task.id, out["success"], out["violation"], 2, extra=out))
    assert [r.violation for r in results] == [False, True, False]
    assert scenario_goal_completion(results) == 0.5
    s = summarise(results)
    assert s["TGC"] == pytest.approx(2 / 3) and s["SGC"] == 0.5 and s["LER"] == pytest.approx(1 / 3)


ALFWORLD_DATA = os.environ.get("ALFWORLD_DATA", str(Path.home() / ".cache" / "alfworld"))


@pytest.mark.skipif(not Path(ALFWORLD_DATA, "json_2.1.1").exists(), reason="ALFWorld data not downloaded")
def test_alfworld_expert_plan_succeeds_and_only_terminal_invalid_counts():
    pytest.importorskip("alfworld")
    env = create_env("alfworld", data_dir=ALFWORLD_DATA, max_tasks=1, expert=True)
    task = env.tasks("test")[0]
    env.reset(task)
    assert task.goal.startswith("examine") or task.goal
    _, done, info = env.step("fly to the moon")
    assert info["violation"] and not done
    env.reset(task)
    for _ in range(50):
        _, done, info = env.step(env.expert_plan[0])
        if done:
            break
    out = env.evaluate()
    assert out["success"] and not out["violation"]
    env.close()


@pytest.mark.skipif(not Path(ALFWORLD_DATA, "json_2.1.1").exists(), reason="ALFWorld data not downloaded")
def test_weva_loop_runs_on_alfworld():
    pytest.importorskip("alfworld")
    from weva import load_config
    from weva.agent import WEvaAgent
    from weva.llm import ScriptedLLM

    llm = ScriptedLLM(
        {
            "perceptor": lambda p: "V: agent in a room\nC: none\nE: progress: 0 of 1",
            "world_model": lambda p: "V: agent in a room\nC: none\nE: progress: 0 of 1\nVIOLATED: none",
            "actor": lambda p: "look\ninventory",
            "controller": lambda p: "LOOKAHEAD",
            "precondition": lambda p: "PRECONDITION: none",
            "verifier": lambda p: "NO",
        }
    )
    env = create_env("alfworld", data_dir=ALFWORLD_DATA, max_tasks=1)
    cfg = load_config(["configs/methods/weva.yaml", "configs/toy.yaml"])
    res = run_episode(WEvaAgent(cfg, llm), env, env.tasks("test")[0], max_steps=3)
    assert res.env_steps == len(res.transitions) == 3
    env.close()
