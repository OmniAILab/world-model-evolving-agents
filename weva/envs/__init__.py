from .base import Env, Task
from .toy_ledger import ToyLedgerEnv
from .wrappers import DynamicsShift

ENV_REGISTRY = {
    "toy-ledger": "weva.envs.toy_ledger.ToyLedgerEnv",
    "alfworld": "weva.envs.adapters.ALFWorldEnv",
    "scienceworld": "weva.envs.scienceworld.ScienceWorldEnv",
    "webshop": "weva.envs.adapters.WebShopEnv",
    "appworld": "weva.envs.adapters.AppWorldEnv",
    "worldbench": "weva.envs.worldbench.WorldBenchEnv",
}


def create_env(name: str, **kwargs) -> Env:
    import importlib

    path = ENV_REGISTRY.get(name, name).replace(":", ".")
    mod, cls = path.rsplit(".", 1)
    return getattr(importlib.import_module(mod), cls)(**kwargs)


__all__ = ["ENV_REGISTRY", "DynamicsShift", "Env", "Task", "ToyLedgerEnv", "create_env"]
