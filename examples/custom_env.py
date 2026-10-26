from __future__ import annotations

import argparse
from typing import Any

from weva import build_agent, load_config, run_benchmark, summarise
from weva.envs import Env, Task


class VaultEnv(Env):
    name = "vault"

    def tasks(self, split: str = "test") -> list[Task]:
        return [Task(id="vault-0", goal="Open the vault and take the document.")]

    def reset(self, task: Task) -> str:
        self.has_key = False
        self.vault_open = False
        self.has_document = False
        self.violated = False
        return self._obs()

    def _obs(self, note: str = "") -> str:
        return (
            "ROOM: office. Objects: key (on desk), vault (locked unless opened with the key).\n"
            f"INVENTORY: {'key' if self.has_key else 'empty'}\n"
            f"VAULT: {'open' if self.vault_open else 'closed'}\n"
            f"NOTE: {note or 'none'}"
        )

    def step(self, action: str) -> tuple[str, bool, dict[str, Any]]:
        a = action.lower()
        info: dict[str, Any] = {"violation": False}
        note = "nothing happened"
        if "take key" in a:
            self.has_key, note = True, "you took the key"
        elif "open vault" in a:
            if self.has_key:
                self.vault_open, note = True, "vault opened"
            else:
                self.violated = info["violation"] = True
                note = "ERROR: the vault cannot be opened without the key"
        elif "take document" in a and self.vault_open:
            self.has_document, note = True, "you took the document"
        return self._obs(note), self.has_document, info

    def evaluate(self) -> dict[str, Any]:
        return {"success": self.has_document, "violation": self.violated}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="gemini/gemini-3.5-flash")
    p.add_argument("--config", default="configs/methods/weva_doc.yaml")
    args = p.parse_args()
    cfg = load_config(args.config, ["llm.backend=litellm", f"llm.model={args.model}"])
    print(summarise(run_benchmark(build_agent(cfg), VaultEnv(), max_steps=cfg.run.max_steps)))
