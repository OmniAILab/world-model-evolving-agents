from __future__ import annotations

import glob
import os
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any, ClassVar

from .base import Env, Task


def _patch_textworld_eval() -> None:
    from textworld.envs.pddl import textgen

    if getattr(textgen.EvalSymbol, "_weva_patched", False):
        return

    def derive(self, context=None):
        context = context or self.context
        value = eval(self.expression, vars(textgen), dict(context["variables"]))
        return [textgen.TerminalSymbol(value)]

    textgen.EvalSymbol.derive = derive
    textgen.EvalSymbol._weva_patched = True


class ALFWorldEnv(Env):
    name = "alfworld"
    SPLITS: ClassVar[dict] = {
        "train": "train",
        "calibration": "valid_seen",
        "dev": "valid_seen",
        "test": "valid_unseen",
    }

    def __init__(
        self,
        data_dir: str | None = None,
        max_tasks: int | None = None,
        task_types: list[str] | None = None,
        step_limit: int = 50,
        show_admissible: bool = False,
        expert: bool = False,
    ):
        try:
            import textworld
            import textworld.gym
            from alfworld.agents.environment.alfred_tw_env import (
                AlfredDemangler,
                AlfredExpert,
                AlfredExpertType,
                AlfredInfos,
            )
        except ImportError as e:
            raise ImportError(
                "ALFWorld is not installed: pip install -e '.[alfworld]' && alfworld-download"
            ) from e
        _patch_textworld_eval()
        self._textworld = textworld
        self._wrappers = [AlfredDemangler(), AlfredInfos]
        if expert:
            self._wrappers.append(AlfredExpert(AlfredExpertType.HANDCODED))
        self.data_dir = Path(data_dir or os.environ.get("ALFWORLD_DATA", Path.home() / ".cache" / "alfworld"))
        self.max_tasks = max_tasks
        self.task_types = task_types
        self.step_limit = step_limit
        self.show_admissible = show_admissible
        self.expert = expert
        self.env = None

    def tasks(self, split: str = "test") -> list[Task]:
        folder = self.SPLITS.get(split, split)
        games = sorted(glob.glob(str(self.data_dir / "json_2.1.1" / folder / "*" / "*" / "game.tw-pddl")))
        if self.task_types:
            games = [g for g in games if Path(g).parent.parent.name.split("-")[0] in self.task_types]
        if self.max_tasks is not None:
            games = games[: self.max_tasks]
        out = []
        for g in games:
            trial = Path(g).parent
            kind = trial.parent.name.split("-")
            out.append(
                Task(id=f"{trial.parent.name}/{trial.name}", goal=" ".join(kind[:-1]), metadata={"game": g})
            )
        return out

    def reset(self, task: Task) -> str:
        tw = self._textworld
        if self.env is not None:
            self.env.close()
        infos = tw.EnvInfos(won=True, admissible_commands=True, extras=["gamefile", "expert_plan"])
        env_id = tw.gym.register_games(
            [task.metadata["game"]],
            infos,
            batch_size=1,
            asynchronous=False,
            max_episode_steps=self.step_limit,
            wrappers=self._wrappers,
        )
        self.env = tw.gym.make(env_id)
        obs, info = self.env.reset()
        text = obs[0]
        m = re.search(r"Your task is to:\s*(.+)", text)
        if m:
            task.goal = m.group(1).strip()
        self.task = task
        self.won = False
        self.last_invalid = False
        self.n_invalid = 0
        self._update(info)
        return self._render(text)

    def _update(self, info: dict) -> None:
        self.admissible = list(info["admissible_commands"][0])
        self.expert_plan = list(info.get("extra.expert_plan", [[]])[0] or [])

    def _render(self, text: str) -> str:
        if self.show_admissible:
            text += "\nADMISSIBLE ACTIONS: " + "; ".join(self.admissible)
        return text

    def step(self, action: str) -> tuple[str, bool, dict[str, Any]]:
        action = action.strip()
        invalid = action not in self.admissible
        obs, _, done, info = self.env.step([action])
        self.won = bool(info["won"][0])
        self.last_invalid = invalid
        self.n_invalid += int(invalid)
        self._update(info)
        return (
            self._render(obs[0]),
            bool(done[0]) or self.won,
            {
                "violation": invalid,
                "invalid_action": invalid,
                "progress": float(self.won),
            },
        )

    def evaluate(self) -> dict[str, Any]:
        return {
            "success": self.won,
            "violation": (not self.won) and self.last_invalid,
            "invalid_actions": self.n_invalid,
        }

    def close(self) -> None:
        if self.env is not None:
            self.env.close()
            self.env = None


class WebShopEnv(Env):
    name = "webshop"
    SPLITS: ClassVar[dict] = {
        "test": (0, 500),
        "dev": (500, 1500),
        "calibration": (500, 1500),
        "train": (1500, None),
    }

    def __init__(
        self,
        env: Any = None,
        num_products: int | None = None,
        max_tasks: int | None = None,
        show_actions: bool = True,
        human_goals: bool = True,
    ):
        if env is None:
            try:
                from web_agent_site.envs.web_agent_text_env import WebAgentTextEnv
            except ImportError as e:
                raise ImportError(
                    "WebShop is not installed: clone princeton-nlp/WebShop, run its setup.sh, "
                    "and start from its root"
                ) from e
            env = WebAgentTextEnv(observation_mode="text", num_products=num_products, human_goals=human_goals)
        self.env = env
        self.max_tasks = max_tasks
        self.show_actions = show_actions

    def tasks(self, split: str = "test") -> list[Task]:
        lo, hi = self.SPLITS[split]
        hi = hi if hi is not None else len(self.env.server.goals)
        idx = list(range(lo, hi))
        if self.max_tasks is not None:
            idx = idx[: self.max_tasks]
        return [Task(id=f"webshop-{i}", goal="", metadata={"session": i}) for i in idx]

    def _actions(self) -> dict:
        return self.env.get_available_actions()

    def _render(self, obs: str) -> str:
        if not self.show_actions:
            return obs
        avail = self._actions()
        acts = (["search[<query>]"] if avail["has_search_bar"] else []) + [
            f"click[{c}]" for c in avail["clickables"]
        ]
        return obs + "\nAVAILABLE ACTIONS: " + "; ".join(acts)

    def reset(self, task: Task) -> str:
        obs, _ = self.env.reset(session=task.metadata["session"])
        task.goal = self.env.instruction_text
        self.task = task
        self.reward = 0.0
        self.done = False
        self.n_invalid = 0
        return self._render(obs)

    def _valid(self, action: str) -> bool:
        m = re.match(r"^\s*(search|click)\[(.*)\]\s*$", action, re.IGNORECASE)
        if not m:
            return False
        avail = self._actions()
        kind, arg = m.group(1).lower(), m.group(2).strip().lower()
        if kind == "search":
            return bool(avail["has_search_bar"]) and bool(arg)
        return arg in avail["clickables"] and arg != "search"

    def step(self, action: str) -> tuple[str, bool, dict[str, Any]]:
        invalid = not self._valid(action)
        obs, reward, done, _ = self.env.step(action.strip())
        self.reward = float(reward or 0.0)
        self.done = bool(done)
        self.n_invalid += int(invalid)
        return (
            self._render(obs),
            self.done,
            {
                "violation": invalid,
                "invalid_action": invalid,
                "progress": self.reward if self.done else 0.0,
            },
        )

    def evaluate(self) -> dict[str, Any]:
        verbose = self.env.server.user_sessions.get(self.env.session, {}).get("verbose_info") or {}
        over_budget = self.done and verbose.get("r_price") is False
        return {
            "success": self.done and self.reward >= 1.0,
            "violation": bool(over_budget),
            "score": self.reward,
            "invalid_actions": self.n_invalid,
        }


class AppWorldEnv(Env):
    name = "appworld"
    SPLITS: ClassVar[dict] = {"calibration": "dev", "dev": "dev", "train": "train", "test": "test_normal"}
    PRELUDE = (
        "You act by writing Python code. Call the apps through the `apis` object, for example "
        "apis.api_docs.show_app_descriptions(). Call apis.supervisor.complete_task() when you are done."
    )

    def __init__(
        self,
        world_factory: Callable[[str], Any] | None = None,
        task_ids: dict[str, list[str]] | None = None,
        experiment_name: str = "weva",
        max_tasks: int | None = None,
    ):
        if world_factory is None:
            try:
                from appworld import AppWorld
            except ImportError as e:
                raise ImportError(
                    "AppWorld is not installed: pip install appworld && appworld install && "
                    "appworld download data"
                ) from e

            def world_factory(task_id: str) -> Any:
                return AppWorld(task_id=task_id, experiment_name=experiment_name)

        self.world_factory = world_factory
        self.task_ids = task_ids
        self.max_tasks = max_tasks
        self.world = None

    def tasks(self, split: str = "test") -> list[Task]:
        name = self.SPLITS.get(split, split)
        if self.task_ids is not None:
            ids = list(self.task_ids[name])
        else:
            from appworld import load_task_ids

            ids = load_task_ids(name)
        if self.max_tasks is not None:
            ids = ids[: self.max_tasks]
        return [Task(id=i, goal="", metadata={"scenario_id": i.split("_")[0]}) for i in ids]

    def reset(self, task: Task) -> str:
        self.close()
        self.world = self.world_factory(task.id)
        t = self.world.task
        supervisor = getattr(t, "supervisor", None)
        who = ""
        if supervisor is not None:
            first = getattr(supervisor, "first_name", "")
            last = getattr(supervisor, "last_name", "")
            who = f"SUPERVISOR: {first} {last}".strip()
        task.goal = t.instruction
        self.task = task
        self.n_errors = 0
        return "\n".join(x for x in (f"TASK: {t.instruction}", who, self.PRELUDE) if x)

    @staticmethod
    def _code(action: str) -> str:
        m = re.search(r"```(?:python)?\n(.*?)```", action, re.DOTALL)
        return (m.group(1) if m else action).strip()

    def step(self, action: str) -> tuple[str, bool, dict[str, Any]]:
        out = self.world.execute(self._code(action))
        failed = out.startswith("Execution failed")
        api_error = failed and re.search(r"status code (?:is )?4\d\d|Response status code", out) is not None
        self.n_errors += int(failed)
        done = bool(self.world.task_completed())
        return out, done, {"violation": api_error, "execution_failed": failed}

    def evaluate(self) -> dict[str, Any]:
        tracker = self.world.evaluate()
        failures = list(getattr(tracker, "failures", []))
        collateral = [f for f in failures if f.get("label") == "no_op_pass"]
        return {
            "success": bool(tracker.success),
            "tgc": float(bool(tracker.success)),
            "violation": bool(collateral),
            "scenario_id": self.task.metadata["scenario_id"],
            "pass_count": len(getattr(tracker, "passes", [])),
            "fail_count": len(failures),
            "execution_errors": self.n_errors,
        }

    def close(self) -> None:
        if self.world is not None:
            self.world.close()
            self.world = None

