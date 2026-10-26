from weva import build_agent, create_env, load_config, run_benchmark, summarise

cfg = load_config(["configs/methods/weva.yaml", "configs/toy.yaml"])
agent = build_agent(cfg)
env = create_env("toy-ledger")
results = run_benchmark(agent, env, max_steps=cfg.run.max_steps)
print(summarise(results))

first = results[0].records[0]
print(
    f"meta-action={first.meta_action}, expanded={first.n_expanded}, pruned={first.n_pruned}, "
    f"changed the Actor's default action: {first.lookahead_changed_action}"
)
