import json
import os
from pathlib import Path

import pytest

ROOT = os.environ.get("WORLDBENCH_ROOT")
pytestmark = pytest.mark.skipif(
    not ROOT or not Path(ROOT, "run_benchmark.py").is_file(),
    reason="set WORLDBENCH_ROOT to a WorldBench checkout",
)


def _bench(tmp_path):
    persona = tmp_path / "bench" / "IT" / "Persona_1"
    (persona / "subtasks").mkdir(parents=True)
    data = persona / "testbed" / "data"
    data.mkdir(parents=True)
    (data / "registro.txt").write_text("riga 4: ok\n")
    (data / "registro_v2.txt").write_text("riga 4: ok\n")
    task = {
        "persona": "Chiara, biologa marina a Chioggia.",
        "task": "Scrivi 'eccezione_MOSE' in ./data/registro.txt.",
        "evaluation": [
            {
                "function": "evaluate_contain",
                "args": {"doc_type": "txt", "file": "./data/registro.txt", "keywords": ["eccezione_MOSE"]},
            }
        ],
    }
    (persona / "subtasks" / "0.json").write_text(json.dumps(task))
    return tmp_path / "bench"


def _shell(cmd):
    return json.dumps({"app": "shell", "action": "command", "command": cmd})


def _env(tmp_path):
    from weva.envs import create_env

    return create_env(
        "worldbench", root=ROOT, benchmark_dir=str(_bench(tmp_path)), work_dir=str(tmp_path / "out")
    )


def test_tasks_carry_the_action_space_and_reset_lists_the_files(tmp_path):
    env = _env(tmp_path)
    task = env.tasks()[0]
    assert task.id == "IT/Persona_1/0" and "switch_app" in task.goal and "finish_task" in task.goal
    obs = env.reset(task)
    assert "registro.txt" in obs and "registro_v2.txt" in obs and "CURRENT APP: none" in obs
    with pytest.raises(ValueError):
        env.tasks("train")


def test_cts_progress_and_step_level_violation_come_from_worldbench(tmp_path):
    env = _env(tmp_path)
    task = env.tasks()[0]
    env.reset(task)
    switch = json.dumps({"app": "system", "action": "switch_app", "target_app": "shell"})
    _, _, info = env.step(switch)
    assert not info["violation"] and info["progress"] == 0.0
    _, _, info = env.step(_shell("echo eccezione_MOSE >> /testbed/data/registro.txt"))
    assert not info["violation"] and info["progress"] == 1.0
    _, done, _ = env.step(json.dumps({"app": "system", "action": "finish_task", "answer": "None"}))
    out = env.evaluate()
    assert done and out["tgc"] == 1.0 and out["cts"] == 1.0 and not out["violation"]

    env.reset(task)
    env.step(switch)
    _, _, info = env.step(_shell("echo eccezione_MOSE >> /testbed/data/registro_v2.txt"))
    assert info["violation"]
    env.step(_shell("echo eccezione_MOSE >> /testbed/data/registro.txt"))
    out = env.evaluate()
    assert out["tgc"] == 1.0 and out["cts"] == 0.0 and out["violation"] and out["n_modified"] == 1


def test_weva_loop_runs_on_worldbench(tmp_path):
    from weva import build_agent, load_config
    from weva.evaluation import run_episode
    from weva.llm import ScriptedLLM

    plan = [
        json.dumps({"app": "system", "action": "switch_app", "target_app": "shell"}),
        _shell("echo eccezione_MOSE >> /testbed/data/registro.txt"),
        json.dumps({"app": "system", "action": "finish_task", "answer": "None"}),
    ]
    seen = []

    def actor(prompt):
        seen.append(prompt)
        return plan[min(len(seen) - 1, 2)]

    llm = ScriptedLLM(
        {
            "perceptor": lambda p: "V: registro.txt\nC: registro_v2.txt must not change\nE: progress: 0 of 1",
            "world_model": lambda p: (
                "V: registro.txt\nC: registro_v2.txt must not change\nE: progress: 1 of 1\nVIOLATED: none"
            ),
            "actor": actor,
            "verifier": lambda p: "NO",
            "controller": lambda p: "LOOKAHEAD",
            "precondition": lambda p: "PRECONDITION: none",
        }
    )
    cfg = load_config(["configs/methods/weva_doc.yaml"], ["llm.backend=toy", "lookahead.horizon=1"])
    env = _env(tmp_path)
    res = run_episode(build_agent(cfg, llm), env, env.tasks()[0], max_steps=3)
    assert res.env_steps == 3 and not res.violation
    assert res.extra["tgc"] == 1.0 and res.extra["cts"] == 1.0
