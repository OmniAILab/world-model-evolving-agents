# WEvA: World-Model Evolving Agents

WEvA is a framework for language agents that predict the consequences of an action before taking it. Most agents
that learn from experience use it to decide which action to propose next, and they find out what that action did
only after it has been executed, when a mistake over a long horizon may no longer be reversible. WEvA turns
experience into a predictive model instead. From its own interaction traces it learns which information is worth
keeping in the state and how actions change that state, using a quasi-symbolic representation built around
entities, preconditions and outcomes.

Before acting, the agent rolls candidate action sequences forward through the learned dynamics, discards those
whose predicted transitions violate an active precondition, and executes the first action of the best branch that
survives. The agreement between predicted and realised states gives a model-trust signal, which decides when
looking ahead is reliable and when the agent should rather interact with the environment and repair its world
model. The repository contains the agent, the two training phases, the adapters for ALFWorld, ScienceWorld,
WebShop, AppWorld and WorldBench, and the configurations of the variants compared in the paper.

**Paper:** [World-Model Evolving Agents](https://aclanthology.org/2026.emnlp-main.2167/) (Ranaldi, EMNLP 2026).



## Contents

- [How it works](#how-it-works)
- [Configurations](#configurations) · [Running on your own environment](#running-on-your-own-environment)
- [Benchmarks](#benchmarks) · [WorldBench](#worldbench)
- [Gemini runs](#gemini-runs) · [Training](#training)
- [Outputs and metrics](#outputs-and-metrics)
- [Tests](#tests) · [Project layout](#project-layout)
- [Citation](#citation)

## How it works

Four components share one quasi-symbolic state σ = (V, C, E), where V holds the entities and their current
values, C the preconditions that constrain the next actions, and E the progress towards the goal.

| Component | What it does | Code |
| --- | --- | --- |
| Perceptor | induces σ_t from the goal, the observation and the recent interaction | `components.Perceptor` |
| World Model | predicts the successor σ̂_{t+1} = F(σ_t, a) and marks violated preconditions | `components.WorldModel` |
| Actor | proposes B candidate actions from σ_t | `components.Actor` |
| Controller | chooses a meta-action from confidence, contradiction, novelty, progress and model trust | `components.Controller` |

At every step (`WEvaAgent.act`):

1. The Perceptor induces σ_t from the goal, the current observation and the last k actions with the observations they produced. Preconditions promoted in memory are always part of C_t.
2. The prediction made for the previous action is compared with σ_t. The agreement updates the model trust μ_t, the transition is stored, and a disagreement that keeps recurring can become a candidate precondition.
3. When trust falls below its alert level, the field with the lowest agreement (V, C or E) is re-induced.
4. The Controller chooses one of `LOOKAHEAD`, `ACT`, `ABSTRACT`, `RETRIEVE` and `ROLLBACK`.
5. With `LOOKAHEAD`, the agent expands B candidates at each predicted state, keeps the K best branches up to depth H, prunes every branch with a predicted violation, and executes the first action of the branch with the highest predicted progress. If no branch survives, or another meta-action was chosen, the Actor's own first proposal is executed without verification.
6. The prediction for the executed action is kept for the comparison at the next step. Predictions for branches that were not executed never enter the trust estimate.

Memory has three levels. The episodic level stores realised transitions, the semantic level records what each
kind of action tends to change, and the procedural level holds the preconditions used during action selection. A
candidate precondition is promoted between episodes, and only if it rejects at least one observed failure without
contradicting any earlier successful transition.

## Configurations

Each variant of the paper is a YAML file in `configs/methods/`:

| Config | Variant |
| --- | --- |
| `weva.yaml` | WEvA with trainable weights (Qwen3.5-9B, Phase I and Phase II) |
| `weva_doc.yaml` | WEvA-Doc: prompted components on Gemini, frozen weights |
| `weva_h0.yaml` | transition monitoring without prospective search |
| `weva_u.yaml` | lookahead without verification |
| `weva_nt.yaml` | lookahead without model-trust gating |
| `weva_f.yaml` | world model and preconditions fixed: no correction, no promotion |
| `free_text_pred.yaml` | free-text state prediction in place of V/C/E |
| `react.yaml` | ReAct baseline |

Several files can be passed to `--config`, and later files override earlier ones. Give the method first and a backbone overlay after it: `configs/backbones/gemini.yaml` for Gemini, or `configs/toy.yaml` for the offline scripted model. Method files change only the keys they name `--config configs/methods/weva_h0.yaml
configs/backbones/gemini.yaml` runs WEvA-H0 on Gemini.

Any setting can also be changed on the command line, for instance `--set lookahead.horizon=3 controller.alert=0.6`, `--set memory.levels=[] memory.retrieval_policy=none` to remove
memory, or `--set correction.enabled=false` to stop world-model correction. The two training-time variants, without transition learning and without the predictability reward for the Perceptor, are Phase II configs in `configs/training/ablations/`.

To run every method config on one environment and print a summary table:

```bash
bash scripts/run_methods.sh configs/envs/toy-ledger.yaml
```

## Running on your own environment

An environment is a class with four methods:

```python
from weva.envs import Env, Task

class MyEnv(Env):
    name = "my-env"

    def tasks(self, split="test") -> list[Task]: ...
    def reset(self, task: Task) -> str: ...
    def step(self, action: str) -> tuple[str, bool, dict]: ...
    def evaluate(self) -> dict: ...
```

| Method | Contract |
| --- | --- |
| `tasks(split)` | list of `Task(id, goal, metadata)` |
| `reset(task)` | starts the task and returns the first observation as text |
| `step(action)` | returns `(observation, done, info)`; `action` is the text the agent produced |
| `evaluate()` | returns `{"success": bool, "violation": bool}` at the end of the episode, optionally `"score"` in [0, 1] |

`info["violation"]` has to come from the environment's own checker: an illegal action, a rejected API call, a file that should not have changed. It is the signal behind LER and the failure labels used to acquire preconditions, so it should never be produced by an LLM. When the environment can measure progress, report it in [0, 1] as `info["progress"]`; the Controller uses it as ℓ_t, which otherwise stays at 0. Observations should be
self-contained text. A complete example is in [`examples/custom_env.py`](examples/custom_env.py).

From the command line the environment can be any importable `module:Class`:

```bash
PYTHONPATH=examples weva run --config configs/methods/weva_doc.yaml --env custom_env:VaultEnv --name vault
```

Constructor arguments go in an env config (`env: module:Class` and `kwargs: {...}`) passed with `--env-config`.
From Python:

```python
from weva import build_agent, load_config, run_benchmark, summarise

cfg = load_config("configs/methods/weva_doc.yaml")
results = run_benchmark(build_agent(cfg), MyEnv(), max_steps=cfg.run.max_steps, out_path="episodes.jsonl")
print(summarise(results))
```

`weva.envs.DynamicsShift` wraps any environment and introduces an unannounced change after a fixed number of
cumulative steps, as in ScienceWorld-Shift. `scripts/sw_shift.py` shows how to read the trust series and the
detection lead from such a run.

## Benchmarks

| Benchmark | `--env` | Install | Metrics |
| --- | --- | --- | --- |
| ALFWorld | `alfworld` | `pip install -e ".[alfworld]"`, then `alfworld-download` | success |
| ScienceWorld | `scienceworld` | `pip install -e ".[scienceworld]"` (needs Java) | success, score |
| ScienceWorld-Shift | `DynamicsShift` around `scienceworld` | | success |
| WebShop | `webshop` | the WebShop repository and its `setup.sh` | score |
| AppWorld | `appworld` | `pip install -e ".[appworld]"`, `appworld install`, `appworld download data` | TGC, SGC |
| WorldBench | `worldbench` | a clone of [OmniAILab/WorldBench](https://github.com/OmniAILab/WorldBench) | TGC, CTS |

Every adapter reports violations at two levels. `info["violation"]` marks a single action that the environment
rejects, and feeds precondition acquisition. `evaluate()["violation"]` marks a trajectory that ends in an
externally verified violation, which is what Avg.LER counts.

| Benchmark | Step-level violation | Terminal violation (Avg.LER) |
| --- | --- | --- |
| ALFWorld | command outside the admissible commands | the episode ends on an inadmissible command without success |
| ScienceWorld | action rejected by the simulator | the simulator fails the task (score −100) |
| WebShop | neither a valid search nor an available click | the purchase exceeds the price limit of the instruction |
| AppWorld | API call rejected with a 4xx status | a "no-op pass" test fails, i.e. the code changed state it should have left alone |
| WorldBench | an action modifies or deletes a pre-existing non-target file | the preservation check fails at the end (CTS = 0) |

The splits are `train`, `dev` (also used as `calibration` in Phase I) and `test`, mapped to each benchmark's
standard splits: `valid_unseen` is the ALFWorld test split, WebShop goals 0–499 are the test split and 500–1499
the dev split, and the AppWorld test split is `test_normal`. Constructor options go in `configs/envs/`.

### WorldBench

WorldBench is not part of this repository. The adapter (`weva/envs/worldbench.py`) loads it from a local clone and
uses its sandbox, its apps and JSON action format, its evaluation functions and its preservation check, so TGC
and CTS are computed exactly as by WorldBench's own runner. WorldBench has a single test split.

```bash
git clone https://github.com/OmniAILab/WorldBench
pip install -r WorldBench/requirements.txt
export WORLDBENCH_ROOT=$PWD/WorldBench
```

The goal given to the agent contains the persona, the instruction and the action space. The first observation
lists the files in `/testbed/data`, and every later one reports the active app and the result of the last action.
`info["progress"]` is the share of the task's deterministic evaluators that already pass; the LLM-judged ones
(`evaluate_email`, `evaluate_note`) are scored only at the end, by WorldBench's judge. Each task's sandbox and
`env_history.json` are kept under `outputs/worldbench/<language>/<persona>/<id>/`.

`configs/envs/worldbench.yaml` runs the whole benchmark and lists every option of the adapter:

| Key | Meaning |
| --- | --- |
| `root` | the WorldBench clone; `null` reads `WORLDBENCH_ROOT` |
| `benchmark_dir` | the task tree to scan; `null` uses `<root>/data` |
| `languages` | settings to run (`DE`, `EN-UK`, `EN-US`, `ES`, `FR`, `IT`, `PT`, `ZH`) |
| `personas`, `subtasks` | restrict to these persona folders and sub-task ids; `null` keeps them all |
| `max_tasks` | keep only the first N tasks after the other filters |
| `max_obs_chars` | truncate each observation to N characters (0 keeps it whole, as in WorldBench) |
| `skip_judged` | leave out tasks scored by the LLM judge |
| `work_dir` | where sandboxes and trajectories are written |

`configs/envs/worldbench_demo.yaml` selects a single Italian task, `IT/Persona_1/0`. The agent has to fill the
`Nota_Eccezione` column of `registro_laguna_chioggia.xlsx`, and `registro_laguna_chioggia_v2.xlsx`, which looks the
same, must stay untouched.

```bash
ENV_CFG=configs/envs/worldbench_demo.yaml bash scripts/run_worldbench.sh --verbose
```

Without `ENV_CFG` the script runs the whole benchmark with WEvA-Doc.

## Gemini runs

API runs use Gemini through [LiteLLM](https://github.com/BerriAI/litellm), with `gemini/gemini-3.5-flash` as the
default model (`configs/backbones/gemini.yaml`; `weva_doc.yaml` uses it too). Another model can be set with
`--set llm.model=...`. The overlay also selects the prompted Controller of Appendix I, so every WEvA variant run on
Gemini shares the Controller of WEvA-Doc.

```bash
pip install -e ".[llm]"
export GEMINI_API_KEY="<your-api-key>"
bash scripts/run_gemini.sh configs/envs/alfworld.yaml
```

The script runs ReAct, free-text state prediction, WEvA-F and WEvA-Doc with the same backbone, split and budget,
then prints a summary table. EvA, EvA-Doc, EvA-M and WorldEvolver are not included; they can be added by wrapping
their code in the `Agent` protocol of `weva/baselines.py`.

A few details matter with Gemini 3.x models:

- The paper's decoding settings (temperature 0, Actor temperature 0.3, top-p 0.9) are sent to the API. Google
  recommends leaving them at their defaults for these models; `--set llm.send_sampling_params=false` does that, but
  the decoding then differs from the paper's.
- These models think before answering, and the thinking counts towards the output budget. Short outputs, such as
  the verifier's YES/NO or the Controller's meta-action, are therefore requested with at least
  `llm.min_output_tokens` tokens (4096).
- The API does not return token log-probabilities for every model, so the Gemini configs set
  `llm.logprobs: false`. The Perceptor's confidence c_t then falls back to 1 / (number of Actor candidates).
- Anything in `llm.extra` is passed to LiteLLM unchanged (`num_retries: 3` by default).

`--verbose` prints one line per executed action, with the meta-action, the model trust, the number of pruned
branches, the action and the observation.

## Training

WEvA comes in two configurations, as in the paper:

| Config | Weights | LLM | Controller | Learning |
| --- | --- | --- | --- | --- |
| `weva.yaml` | trainable | Qwen3.5-9B through `TorchBackend` | learned in Phase I | Phase I, Phase II, memory between episodes |
| `weva_doc.yaml` | frozen | Gemini through LiteLLM | prompted (Appendix I) | memory between episodes only |

`weva/torch_backend.py` loads the backbone with transformers and attaches one LoRA adapter per operation
(Perceptor, World Model, Actor) to the shared, frozen backbone. The Perceptor and the Actor are updated with GRPO
(group-normalised advantages, clipped ratio, k3 KL to the backbone without adapters), and the World Model with the
negative log-likelihood L_W of the realised successor, `VIOLATED:` line included. Only the adapter being updated
has `requires_grad`, each adapter has its own optimiser, and the trainable set is checked before every update. The
same backend serves the agent during rollouts, so the adapters that are trained are the ones that act. Install
with `pip install -e ".[train]"`.

**Phase I** calibrates the Controller on post-adaptation performance. For each calibration task (50, as in EvA),
an agent with a sampled Controller adapts for five rounds; each round collects a group of rollouts and, for
trainable WEvA, applies a Phase II update. The agent then plays an evaluation episode, and its result rewards the
Controller through a score-function policy gradient. The weights are restored after each task.

```bash
weva train-controller --config configs/methods/weva.yaml --env-config my_env.yaml --backend torch \
  --training configs/training/phase1.yaml --out outputs/controller.json
```

**Phase II** adapts the other components with the Controller fixed. In each round:

1. G = 8 rollouts are collected per training task, all starting from the same memory, Controller state and
   weights.
2. With the World Model frozen, the Perceptor is updated with GRPO on G sampled inductions of each source state
   whose confidence passes the EvA gate (c ≥ 0.6). The reward is EvA's R_task + α·R_cons + β·R_nov plus the
   predictability term log p(σ_{t+1} | σ_t, a_t).
3. Source and successor states are re-induced with the updated Perceptor.
4. With the Perceptor frozen, the World Model is updated on the re-induced transitions with L_W.
5. The Actor is updated with GRPO on the candidate lists it generated; the rollout's advantage goes to the
   candidate that was executed.

```bash
weva train-phase2 --config configs/methods/weva.yaml --env-config my_env.yaml --backend torch \
  --controller outputs/controller.json --training configs/training/phase2.yaml
```

The adapters are saved to `outputs/phase2/adapters/`, which `weva.yaml` loads for evaluation. With
`--backend null` the same loop runs without changing any weight, which is a cheap way to check an environment.
The backend is a single-process PyTorch implementation; the experiments in the paper used VeRL. SLURM scripts for
both phases are in `scripts/slurm/`.

## Outputs and metrics

Every `weva run` writes to `outputs/<name>/`:

- `run.json`: the resolved config, the overrides, the environment and its arguments, the split, the task ids, the
  seed, the step budget and the git commit.
- `episodes.jsonl`: one line per episode, with success, terminal violation, executed steps, lookahead expansions,
  LLM calls and the full step trace.
- `summary.json`: the metrics below, averaged over the run.

| Metric | Meaning |
| --- | --- |
| success | share of episodes that reach the goal |
| LER | share of episodes that end in a verified violation |
| TGC, CTS, SGC, score | reported when the benchmark provides them |
| TF | transition fidelity between predicted and realised states, overall and per field (V, C, E) |
| detection lead | steps between model trust crossing its alert level and task success degrading |

Only executed environment actions count as interaction steps. Lookahead expansions are predicted transitions that
were never executed, and the LLM-call count includes every call made by the Perceptor, World Model, Actor,
verifier, precondition proposer and Controller. To compare methods, keep the split, backbone, `run.max_steps` and
seed fixed; `run.json` records all of them. `python scripts/summarise_runs.py outputs` collects the summaries
into one table.


## Tests

```bash
pip install -e ".[dev]"
pytest -q
```

The tests run offline with the scripted model and cover the state, lookahead and verification, model trust,
memory and promotion, the Controller, both training phases and the adapters. The training-backend tests run when
`.[train]` is installed, the ScienceWorld tests when ScienceWorld and Java are available, and the WorldBench tests
when `WORLDBENCH_ROOT` points to a clone.

## Project layout

```
world-model-evolving-agents/
  weva/
    types.py          # State, Transition, ControlState, MetaAction
    config.py         # typed config, YAML loading, key=value overrides
    state.py          # V/C/E parsing, similarity sim_ψ, progress readout
    llm.py            # LLM interface, scripted model, LiteLLM backend
    components.py     # Perceptor, World Model, Actor, Controller
    search.py         # verified lookahead
    trust.py          # model trust and field-level diagnosis
    memory.py         # episodic, semantic and procedural memory, precondition promotion
    agent.py          # WEvaAgent and build_agent
    baselines.py      # ReAct and the Agent protocol for external baselines
    evaluation.py     # metrics, run_episode, run_benchmark, summarise
    grpo.py           # group advantages, clipped objective, KL, successor NLL
    torch_backend.py  # adapters, GRPO and L_W updates
    training.py       # EvA rewards, Phase I calibration, Phase II loop
    cli.py            # weva demo | run | train-controller | train-phase2
    envs/             # ledger, ALFWorld, ScienceWorld, WebShop, AppWorld, WorldBench, DynamicsShift
    prompts/          # the prompts 
  configs/            # default.yaml, methods/, backbones/, envs/, training/, toy.yaml
  scripts/            # run_methods.sh, run_gemini.sh, run_worldbench.sh, sw_shift.py, summarise_runs.py, slurm/
  examples/           # quickstart.py, custom_env.py
  tests/
```

## Citation

If you use WEvA in your research, please cite:

```bibtex
@inproceedings{ranaldi-2026-world,
    title = "World-Model Evolving Agents",
    author = "Ranaldi, Leonardo",
    editor = "Lee, Hung-Yi  and
      Dev, Sunipa  and
      Li, Jessy  and
      Elliott, Desmond",
    booktitle = "Proceedings of the 2026 Conference on Empirical Methods in Natural Language Processing",
    month = oct,
    year = "2026",
    address = "Budapest, Hungary",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2026.emnlp-main.2167/",
    pages = "49279--49294",
    ISBN = "979-8-89176-459-0"
}
```

Released under the MIT License.
