import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from weva import build_agent, create_env, load_config
from weva.evaluation import detection_lead, first_crossing, run_episode

cfg = load_config(["configs/methods/weva.yaml", "configs/toy.yaml"], ["controller.alert=0.8"])
agent = build_agent(cfg)
env = create_env("toy-ledger", shift_after=6)
trust_series, success_series = [], []
last_success = 1.0
for i in range(8):
    task = env.tasks()[i % 3]
    res = run_episode(agent, env, task, max_steps=8)
    recs = res.records[1:]
    trust_series += [s.trust for s in recs]
    success_series += [last_success] * (len(recs) - 1) + [float(res.success)]
    last_success = float(res.success)

alert, degradation = cfg.controller.alert, 0.5
out = {
    "interaction_steps": len(trust_series),
    "trust_alert_step": first_crossing(trust_series, alert),
    "success_degradation_step": first_crossing(success_series, degradation),
    "detection_lead": detection_lead(trust_series, success_series, alert, degradation),
}
print(json.dumps(out, indent=2))
