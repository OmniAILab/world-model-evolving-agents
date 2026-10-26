import pytest

from weva.config import load_config


def test_base_inheritance_and_overrides():
    cfg = load_config("configs/methods/weva_h0.yaml")
    assert cfg.lookahead.horizon == 0 and cfg.lookahead.branching == 3
    cfg = load_config(
        ["configs/methods/weva.yaml", "configs/toy.yaml"], ["lookahead.horizon=3", "controller.alert=0.8"]
    )
    assert cfg.lookahead.horizon == 3 and cfg.controller.alert == 0.8


def test_paper_settings_are_the_defaults():
    cfg = load_config(["configs/methods/weva.yaml", "configs/toy.yaml"])
    assert (cfg.lookahead.horizon, cfg.lookahead.branching, cfg.lookahead.beam) == (2, 3, 2)
    assert (cfg.llm.actor_temperature, cfg.llm.actor_top_p) == (0.3, 0.9)


def test_unknown_key_is_rejected():
    with pytest.raises(KeyError):
        load_config(["configs/methods/weva.yaml", "configs/toy.yaml"], ["lookahead.horizon_typo=1"])


def test_every_shipped_config_loads():
    from pathlib import Path

    for p in Path("configs/methods").glob("*.yaml"):
        load_config(p)


def test_method_overlays_keep_the_backbone():
    cfg = load_config(["configs/methods/weva.yaml", "configs/methods/weva_nt.yaml"])
    assert (cfg.weights, cfg.llm.backend, cfg.controller.type) == ("trainable", "torch", "learned")
    assert cfg.controller.trust_gating is False


def test_backbone_overlay_does_not_undo_a_fixed_controller():
    from weva.components import FixedController, create_controller
    from weva.types import MetaAction

    for overlay in ("configs/toy.yaml", "configs/backbones/gemini.yaml"):
        cfg = load_config(
            ["configs/methods/weva.yaml", overlay], ["controller.trust_gating=false", "controller.fixed_action=ACT"]
        )
        ctrl = create_controller(cfg.controller)
        assert isinstance(ctrl, FixedController) and ctrl.action == MetaAction.ACT
        cfg = load_config(["configs/methods/weva_nt.yaml", overlay])
        assert create_controller(cfg.controller).action == MetaAction.LOOKAHEAD


def test_gemini_variants_share_the_prompted_controller():
    for m in ("weva_f", "weva_h0", "weva_u", "free_text_pred", "weva_doc"):
        cfg = load_config([f"configs/methods/{m}.yaml", "configs/backbones/gemini.yaml"])
        assert (cfg.llm.backend, cfg.controller.type) == ("litellm", "llm")
