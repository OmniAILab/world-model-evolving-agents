import importlib.util
from pathlib import Path

from weva.envs import create_env


def test_create_env_by_import_path():
    env = create_env("weva.envs.toy_ledger:ToyLedgerEnv")
    assert env.name == "toy-ledger"


def test_example_env_follows_contract():
    path = Path(__file__).resolve().parents[1] / "examples" / "custom_env.py"
    spec = importlib.util.spec_from_file_location("custom_env", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    env = mod.VaultEnv()
    task = env.tasks()[0]
    env.reset(task)
    _, done, info = env.step("open vault")
    assert info["violation"] and not done
    env.reset(task)
    env.step("take key")
    env.step("open vault")
    _, done, info = env.step("take document")
    assert done and not info["violation"]
    assert env.evaluate() == {"success": True, "violation": False}
