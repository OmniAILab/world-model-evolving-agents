from __future__ import annotations

from functools import cache
from importlib import resources


@cache
def load(name: str) -> str:
    return resources.files(__package__).joinpath(f"{name}.txt").read_text(encoding="utf-8")


def render(name: str, **kwargs: object) -> str:
    return load(name).format(**kwargs)
