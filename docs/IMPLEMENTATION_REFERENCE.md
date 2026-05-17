# ReiL — Canonical Implementation Reference

> **Purpose.** A factual, repo-grounded inventory of what `ReiL` actually implements today, intended as a companion to the higher-level design docs ([copilot-instructions.md](../copilot-instructions.md), [reil/.instructions.md](../reil/.instructions.md), [docs/ARCHITECTURE.md](ARCHITECTURE.md)). Where those files explain *how to think about ReiL*, this file enumerates the *concrete classes, signatures, defaults, and constants* a developer or agent will encounter in the source tree.
>
> **Scope.** `reil/` package in this repository, snapshotted from `master` (commit `ba3824bf`, "improvements for tandem run"). Where the active development branch `feature/chapters` deviates from master, those deviations are called out explicitly.
>
> **Not a tutorial.** Read this when you need to look something up; use the skills docs and `copilot-instructions.md` for design rationale.
>
> **For paper-side concerns.** The two papers being prepared from this codebase are tracked under `C:\Users\sj_an\Documents\Claude\Projects\Dissertation papers\`. This file does **not** ground claims in the papers; that is the next-step comparison the user will run.

## Table of contents

1. [Package layout](#1-package-layout)
2. [Foundations (`reilbase`, `stateful`, `serialization`, `logger`)](#2-foundations-reilbase-stateful-serialization-logger)
3. [Datatypes (`reil/datatypes/`)](#3-datatypes-reildatatypes)
4. [Buffers (`reil/datatypes/buffers/`)](#4-buffers-reildatatypesbuffers)
5. [Agents (`reil/agents/`)](#5-agents-reilagents)
6. [Learners (`reil/learners/`)](#6-learners-reillearners)
7. [Environments (`reil/environments/`)](#7-environments-reilenvironments)
8. [Subjects (`reil/subjects/` — non-healthcare)](#8-subjects-reilsubjects--non-healthcare)
9. [Utilities (`reil/utils/`)](#9-utilities-reilutils)
10. [Healthcare (`reil/healthcare/`)](#10-healthcare-reilhealthcare)
11. [Legacy (`reil/legacy/`)](#11-legacy-reillegacy)
12. [Top-level package globals (`reil/__init__.py`)](#12-top-level-package-globals-reil__init__py)
13. [`feature/chapters` branch — uncommitted modifications](#13-featurechapters-branch--uncommitted-modifications)
14. [Notes on possible discrepancies](#14-notes-on-possible-discrepancies)

---

## 1. Package layout

```
reil/
├── __init__.py            # Global random generators, FILE_FORMAT, seeding helpers
├── _version.py            # versioneer-generated
├── reilbase.py            # ReilBase (root of class hierarchy)
├── stateful.py            # Stateful (entities with state, statistics, sub-components)
├── serialization.py       # PickleMe, CustomUnPickler, deserialize/serialize, legacy mappings
├── logger.py              # Logger (wrapper around stdlib logging)
├── agents/                # BaseAgent, Agent, AgentDemon, PPO, A2C, DQN, RandomAgent, ...
├── datatypes/             # Feature/FeatureSet, components, entity_register, buffers/, ...
├── environments/          # Environment, Single, Sequential, Task, Session, SessionBuilder
├── healthcare/            # Patient, PK/PD models, Warfarin subjects, baseline agents
├── learners/              # Learner, QLearner, PPOLearner (+ Tandem, Parallel), A2C, LookupTable, ...
├── legacy/                # Deprecated modules (kept for backward-compat with old pickles)
├── subjects/              # Subject, SubjectDemon, FrozenLake, FrozenRiver, TicTacToe, MNKGame
└── utils/                 # yaml_tools, exploration, action_generator, metrics, stopping, ...
```

`pyproject.toml` declares the package as `reil` against Python 3.13 in the Poetry env `reil-l-k_YBAA-py3.13`. Test directory `tests/` contains unit tests.

---

## 2. Foundations (`reilbase`, `stateful`, `serialization`, `logger`)

### 2.1 `ReilBase` ([reil/reilbase.py:19](../reil/reilbase.py#L19))

Root class. Every persistable ReiL object inherits from it.

**Constructor:**
```python
ReilBase(
    name: str | None = None,
    path: pathlib.PurePath | None = None,
    logger_name: str | None = None,
    logger_level: int | None = None,
    logger_filename: str | None = None,
    persistent_attributes: list[str] | None = None,
    save_zipped: bool | None = None,
)
```
- `name` defaults to `self.__class__.__qualname__.lower()`.
- `path` defaults to `'.'`.
- `persistent_attributes` are the attributes preserved across a `load()` (each entry is prefixed with `_` internally).
- `save_zipped`: if `None`, deferred to module-level `reil.FILE_FORMAT` (`'pkl'` by default).

**Key methods:**
- `save(filename=None, path=None) -> pathlib.PurePath` — delegates to `PickleMe.get(fmt).dump(...)`.
- `load(filename, path=None) -> None` — in-place restore via `PickleMe.get(fmt).load(...)`; preserves `_persistent_attributes`.
- `from_pickle(cls, filename, path=None)` — classmethod; creates an empty instance via `_empty_instance()` then loads.
- `from_config(cls, config)` / `get_config()` — config-based serialization round-trip (`internal_states` pocket holds `_persistent_attributes` etc.).
- `reset()` — no-op base; subclasses override.

### 2.2 `Stateful` ([reil/stateful.py:45](../reil/stateful.py#L45))

Adds state management, entity registration, and sub-component discovery on top of `ReilBase`.

**Constructor:**
```python
Stateful(
    min_entity_count: int = 1,
    max_entity_count: int = -1,           # -1 = unlimited
    unique_entities: bool = True,
    state_dumper: FeatureSetDumper | None = None,
    summary_writer: SummaryWriter | None = None,
    **kwargs,
)
```

**Instance attributes set in `__init__`:**
- `state: State` — composed via `_extract_sub_components()` and the subclass's `_state_def_reference` / `_generate_state_defs`.
- `statistic: Statistic` — backed by `self.state`.
- `_entity_list: EntityRegister` — controls how many entities (agents or subjects) can be registered.
- `_metrics: dict[str, MetricProtocol]`, `_computed_metrics: dict[str, float]`.
- `_summary_writer: SummaryWriter | None` — TensorBoard write target (used by subclasses).

**Discovery protocol — `_extract_sub_components()`:**
- Scans `dir(self)` for methods whose name starts with `_sub_comp_`.
- Each such method must accept `(self, _id, **kwargs)`; remaining keyword args become parameterizable inputs to a state definition.
- Returns `dict[str, (callable, tuple_of_param_names)]`, used by `State` to compose state definitions dynamically.

**Two state-definition strategies:**
- `_generate_state_defs()` — eager: definitions live on the instance after `__init__`.
- `_state_def_reference(name)` — lazy: returns the definition tuple for `name` when first requested. Preferred (smaller pickles, single dispatch hit).

**`register(entity_name, _id=None) -> int` / `deregister(entity_id) -> None`** — manage the `EntityRegister`. Used by subjects to enroll agents and vice-versa.

### 2.3 `serialization` ([reil/serialization.py](../reil/serialization.py))

Three responsibilities:

**(a) File-format dispatch — `PickleMe`:**
- Singleton-style registry. `PickleMe.get('pkl')` → `DefaultPickler` (uncompressed `pickle.dump`).
- `PickleMe.get('pbz2')` → `ZippedPickler` (bz2-compressed pickle).
- `dump(obj, filename, path) -> pathlib.PurePath`; `load(filename, path) -> Any` (uses `CustomUnPickler` internally).

**(b) Backward-compat class remapping — `CustomUnPickler`:**
Overrides `pickle.Unpickler.find_class` to swap legacy class paths. The active mappings (examined in source) include:
- `MockStatistic` placeholder swap.
- `PrimaryComponent → State` (the old name for `State`).
- `EnvironmentStaticMap → Sequential` (the old env name).
- `QDense → QLearner` (consolidation of dense Q-net wrapper).
- `FeatureArray → FeatureSet`.
- `FeatureSetDumper` location move.
- Function rename: `change_array_to_missing → change_set_to_missing`.
- Substring rewrite: `interval → duration` (reflects the repo-wide rename committed in `9b141f2`).

If you rename or move a public class, **add to this map** rather than break pre-existing `.pkl` artifacts.

**(c) Config-based (de)serialization — `serialize` / `deserialize`:**
- `full_qualname(obj) -> str` — returns `'module.path>ClassName'`.
- `serialize(obj)` — calls `obj.get_config()`; tags `__needs_deserialization__`.
- `deserialize(object_info)` — looks up the class, calls `from_config()`.

### 2.4 `Logger` ([reil/logger.py](../reil/logger.py))

Lightweight wrapper. Constructor: `Logger(logger_name, logger_level=WARNING, logger_filename=None, fmt=DEFAULT_FORMAT)`.
- Methods: `debug/info/warning/error/exception/critical(msg)`.
- `from_config(config) -> Logger`, `get_config() -> dict`.
- `__getstate__/__setstate__` handle pickle (with old-attribute-name compat).

Top of `reil/logger.py` re-exports constants from `logging` (`DEBUG`, `INFO`, `WARNING`, etc.) for ergonomic use.

---

## 3. Datatypes (`reil/datatypes/`)

### 3.1 `Feature` / `FeatureSet` ([reil/datatypes/feature.py](../reil/datatypes/feature.py))

**`Feature`** (frozen dataclass, [line 35](../reil/datatypes/feature.py#L35)):
- Fields: `name, value, is_numerical, categories, lower, upper, index, normalized, dict_fields`.
- Classmethods: `Feature.numerical(name, value, lower, upper, normalized=None, index=None)`, `Feature.categorical(name, value, categories, normalized=None, index=None)`.
- `as_dict` (cached property) returns the canonical dict view (different fields for numerical vs categorical).
- `__add__` concatenates two compatible Features element-wise.
- `NoneFeature = Feature('None')` — the sentinel returned by `_sub_comp_none`.
- `MISSING = '__missing_feature__'` — sentinel string for "value present but unknown".

**`FeatureGenerator` / `FeatureGeneratorSet`:**
- `FeatureGenerator` produces validated `Feature` instances. Constructors `categorical(name, categories, probabilities=...)`, `numerical(name, lower, upper, mean=..., stddev=...)` etc.
- `FeatureGeneratorSet` aggregates generators and is used by `Patient.feature_gen_set` to sample feature batches.

**`FeatureSet`** — dict-like ordered container of `Feature`s. Provides:
- `.value` (raw dict of names → values).
- `.normalized.flattened` (concatenated normalized representation; what learners consume).
- Iteration / `__getitem__` / `__setitem__` / serialization helpers.

**`FeatureSetDumper`** — protocol for components that persist `FeatureSet`s to disk (see `healthcare/trajectory_dumper.py` for the canonical implementation).

**Helper functions** (free functions):
- `change_set_to_missing(feature_set: FeatureSet) -> FeatureSet` — replaces specified feature values with `MISSING`.

### 3.2 Components ([reil/datatypes/components.py](../reil/datatypes/components.py))

- `State`, `Reward`, `ActionSet`, `Statistic` — all are Component-like dataclasses that take an `object_ref` to the owning `Stateful`, an `available_sub_components` dict (from `Stateful._extract_sub_components`), an optional `dumper`, and a `pickle_stripped` flag.
- `SubComponentInstance(name, kwargs)` — describes a single sub-component invocation in a state definition.
- `SubComponentInfo = tuple[Callable, tuple_of_param_names]`.

### 3.3 Entity registry ([reil/datatypes/entity_register.py](../reil/datatypes/entity_register.py))

- `EntityRegister(min_entity_count, max_entity_count, unique_entities=True)`.
- `append(entity_name, _id=None) -> int` — registers and returns an ID.
- `ready` property — True iff registered count ≥ `min_entity_count`.

### 3.4 Dataclasses re-exports ([reil/datatypes/__init__.py](../reil/datatypes/__init__.py))

Common imports surface through this `__init__`. Notably `Entity`, `InteractionProtocol`, and `History` / `Observation` are exposed as the public datatypes used in environment/protocol code.

### 3.5 `MockStatistic` ([reil/datatypes/mock_statistic.py](../reil/datatypes/mock_statistic.py))

Proxy that intercepts `append(name, _id)` to track history while otherwise delegating to a wrapped `Stateful.statistic`. Used internally by instance generators.

---

## 4. Buffers (`reil/datatypes/buffers/`)

| Class | File | Behavior |
| --- | --- | --- |
| `Buffer[T1, T2]` (base) | `buffer.py` | Abstract container with named queues. Constructor args: `buffer_size`, `buffer_names`, `pick_mode ∈ {'all','random','recent','old'}`, `clear_buffer`. Once `buffer_size`/`buffer_names` are set they cannot be changed. |
| `CircularBuffer[T1, T2]` | `circular_buffer.py` | FIFO ring buffer. New items overwrite oldest when full; `_buffer_full` flag tracked. |
| `VanillaExperienceReplay[T1, T2]` | `vanilla_experience_replay.py` | Extends `CircularBuffer`. Constructor adds `batch_size`; `pick_mode='random'` is forced. Used for off-policy DQN replay. Raises if `batch_size > buffer_size`. |
| `EndlessBuffer[T1, T2]` | `endless_buffer.py` | Unbounded list (`_buffer_size = -1`). Used for on-policy trajectory collection. |
| `FillFlushBuffer[T1, T2]` | `fill_flush_buffer.py` | Extends `EndlessBuffer`. `pick()` returns the whole batch once `buffer_size` is reached, then clears. Used as the canonical PPO buffer. |
| `Sink[T1, T2]` | `sink.py` | No-op buffer. `add()` discards; `pick()` returns empty. Useful as a placeholder when data collection is disabled. |

The `buffer_names` allow heterogeneous parallel queues (e.g., separate `'state'` / `'action'` / `'reward'` tracks). `pick()` returns `dict[name → tuple]`.

---

## 5. Agents (`reil/agents/`)

`reil/agents/__init__.py` re-exports the public agent classes (canonical names below).

### 5.1 `BaseAgent` ([reil/agents/base_agent.py:20](../reil/agents/base_agent.py#L20))

Non-learning baseline. Inherits from `Stateful`.

**Constructor:** `BaseAgent(tie_breaker: Literal['first','last','random']='random', variable_action_count: bool=True, **kwargs)`.

**Key methods:**
- `act(state, subject_id, actions, iteration=0) -> FeatureSet` — abstract surface; subclasses implement decision logic.
- `best_actions(state, actions) -> tuple[FeatureSet, ...]` — raises `NotImplementedError` here.
- `observe(subject_id, stat_name) -> Generator` — coroutine that yields/receives actions and computes statistics at close.
- `_break_tie(input_tuple, method) -> T` (static) — selects from a tuple per `tie_breaker`.

**`_training_trigger`** — `Literal['none','termination','state','action','reward']`. On `BaseAgent` always `'none'`.

### 5.2 `Agent` ([reil/agents/agent.py:27](../reil/agents/agent.py#L27))

Learning agent. Inherits from `BaseAgent`. Generic over `(InputType, LabelType)`.

**Constructor:**
```python
Agent(
    learner: LearnerProtocol[InputType, LabelType],
    exploration_strategy: float | ExplorationStrategy,
    discount_factor: float = 1.0,
    tie_breaker: Literal['first','last','random'] = 'random',
    training_trigger: Literal['none','termination','state','action','reward'] = 'termination',
    **kwargs,
)
```

**Key methods:**
- `act(state, subject_id, actions, iteration=0) -> FeatureSet` — layers exploration on top of `BaseAgent.act`.
- `observe(...) -> Generator` — extends `BaseAgent.observe`; triggers `learn(history)` per `_training_trigger`.
- `learn(history) -> dict[str, float]` — calls `_prepare_training(history)`, dispatches to the learner, updates metrics, returns learner metrics.
- `_prepare_training(history) -> TrainingData[InputType, LabelType]` — abstract; subclasses override.
- `discounted_cum_sum(r, discount) -> list[float]` (static, via `scipy.signal.lfilter`).
- `extract_reward(history, min_clip, max_clip) -> list[float]` (static).
- `get_active_history(history) -> History` (static) — clips terminal state if no action/reward.
- `reset()` — resets the learner if `_training_trigger != 'none'`.

**`_training_trigger` semantics:** `'none'` = pure rollout/eval; `'state'`/`'action'`/`'reward'` = per-event learning; `'termination'` = batch learning when the generator closes (the default and almost always what's used in warfarin experiments).

### 5.3 `AgentDemon` ([reil/agents/agent_demon.py:22](../reil/agents/agent_demon.py#L22))

Wraps another agent and conditionally substitutes a `sub_agent`.

**Constructor:** `AgentDemon(sub_agent: BaseAgent, condition_fn: Callable[[FeatureSet, int], bool], main_agent: BaseAgent | None = None, **kwargs)`.

- `__call__(main_agent)` attaches a main agent and mirrors its state/statistic/entity_list/`_training_trigger` (and on `feature/chapters` also `_summary_writer`, `_learner`, `_computed_metrics`, `_metrics`).
- `act(state, subject_id, actions, iteration)` — dispatches to `sub_agent` when `condition_fn(state, subject_id)` is True, else `main_agent`.
- `learn(history)` — delegates to `main_agent.learn(history)` when `_training_trigger != 'none'`, else returns `{}`. On `feature/chapters` this now returns a dict (previously implicit None).
- `register`/`deregister` register with both wrapped agents (raises `RuntimeError` on ID mismatch).
- `save`/`load` persist both agents to separate files.

### 5.4 `A2CAgent` ([reil/agents/actor_critic_agent.py:24](../reil/agents/actor_critic_agent.py#L24))

Base class for the actor-critic family. `LabelType = tuple[tuple[int|None, ...], float]` (action indices + scalar return).

**Constructor:** `A2CAgent(learner, buffer: Buffer, reward_clip=(None, None), **kwargs)`. Defaults to `NoExploration()` and `variable_action_count=False`.

`_prepare_training(history)` packs `(state, (action_indices, return), {})` through the buffer.

### 5.5 `PPOAgent` ([reil/agents/ppo_agent.py:27](../reil/agents/ppo_agent.py#L27))

Inherits from `A2CAgent`.

**Constructor:** `PPOAgent(learner: PPOLearner, buffer: Buffer, reward_clip=(None, None), gae_lambda: float = 1.0, **kwargs)`.

Key responsibilities:
- `_prepare_training(history)` runs Generalized Advantage Estimation with `gae_lambda` (recommended ≤ 1), packages `(states, (action_indices, return, advantage), {})` into the buffer.
- `_update_metrics(**kwargs)` updates `action_rank`, `advantage_mean/h`, `rewards/h` per minibatch. On `feature/chapters` this is hardened to handle `y` as list or tuple and 1-D tensors via `tf.expand_dims`.

`PPOAgent` is the parent of the warfarin-specific PPO chain in `reil.healthcare.agents.warfarin_ppo_agent` (see §10.5).

### 5.6 Other agents

| Class | File | Constructor highlights |
| --- | --- | --- |
| `DeepQLearningAgent` | `deep_q_agent.py:24` | `(learner: QLearner, buffer, exploration_strategy, default_actions=(), **kwargs)`. Implements `best_actions(state, actions)` with optional lookahead. |
| `QLearningAgent` | `q_learning_agent.py:27` | `(learner, buffer, exploration_strategy, default_actions=(), **kwargs)` — tabular or linear Q-learning. `variable_action_count=True`. |
| `RandomAgent` | `random_agent.py:16` | `(default_actions=(), seed=None, **kwargs)`. Pulls a random feature from `default_actions` or `actions.send('choose feature exclusive')`. No learning. |
| `UserAgent` | `user_agent.py:14` | `(default_actions=(), **kwargs)` — prompts on stdin; useful for human-in-the-loop testing. |
| `TwoPhaseAgent` | `two_phase_agent.py:15` | `(first_agent, second_agent, switch_feature, switch_value, init_state_comps, main_state_comps, **kwargs)`. Switches between two agents based on a state-feature threshold. |
| `LRAgent` | `linear_regression_agent.py:18` | `(default_actions=(), models={0: LinearRegression()}, feature_sequence=(), value_extractor_fn=lambda x: x.value, **kwargs)` — supervised baseline, picks the nearest valid action to a regression prediction. |

---

## 6. Learners (`reil/learners/`)

### 6.1 Base contract

**`LearnerProtocol`** (Protocol, `learner.py`):
- Attributes: `_learning_rate: LearningRateScheduler`, `_iteration: int`.
- Methods: `predict(X, training=False)`, `learn(X, Y) -> dict`, `reset()`, `get_parameters()`, `set_parameters(params)`.

**`Learner`** (`learner.py:85`) — concrete base. Converts a float `learning_rate` to `ConstantLearningRate`; provides `_empty_instance()` classmethod.

### 6.2 Deep Q-Learning ([reil/learners/q_learner.py](../reil/learners/q_learner.py))

- **`DeepQModel`** (Keras model). Constructor: `(learning_rate, validation_split=0.0, hidden_layer_sizes=(1,))`. Implements `call`, `max`, `argmax` (both `@tf.function`), `build`, `fit`, `get_config`, `from_config`.
- **`QLearner`** wraps `DeepQModel`. Constructor: `(model: DeepQModel, tensorboard_path=None, tensorboard_filename=None, **kwargs)`. `learn(X, Y)` returns metrics dict.

`reil/learners/dense.py` re-exports `QLearner` under the historical name `Dense` (used by `warfarin_dosing/wd_utils/experiment.py`).

### 6.3 PPO — three model variants

**`PPOModel`** ([ppo_learner.py:45](../reil/learners/ppo_learner.py#L45)) — vanilla PPO with **independent actor/critic networks**.
Constructor (key args):
```python
PPOModel(
    input_shape,
    action_per_head,
    actor_learning_rate,
    critic_learning_rate,
    actor_layer_sizes,
    critic_layer_sizes,
    actor_train_iterations,
    critic_train_iterations,
    target_kl,
    actor_hidden_activation='relu',
    actor_head_activation=None,
    critic_hidden_activation='relu',
    clip_ratio=None,
    critic_clip_range=None,
    max_grad_norm=None,
    critic_loss_coef=1.0,
    entropy_loss_coef=0.0,
    regularizer_coef=0.0,
)
```
Methods include `train_actor(x, action_indices, advantage)` (`@tf.function`), `train_critic(x, returns)`, `train_step(data)`, `_compute_actor_loss`, `_compute_regularizer_loss`, `_logprobs_j` / `_logprobs_concat`. Supports KL-divergence clipping and entropy regularization.

**`PPONeighborEffect`** ([ppo_learner.py:456](../reil/learners/ppo_learner.py#L456)) — subclass of `PPOModel` adding a "neighbor effect" loss term (`effect_widths`, `effect_decay_factors`, `effect_prob`) that decays gradient on adjacent off-policy actions.

**`PPOTandemModel`** ([ppo_learner_tandem.py:46](../reil/learners/ppo_learner_tandem.py#L46)) — staged training. Key differences vs. `PPOModel`:
- `actor_layer_sizes: dict[str, tuple[int, ...]]` (one entry per backbone name).
- `training_switch: dict[str, int] | None` — toggles which layers train at which iteration.
- `backprop_mode: Literal['separate','shared','all'] = 'all'`.
- `_freeze_layers()` enforces the switch schedule per iteration.

**`PPOParallelModel`** ([ppo_learner_parallel.py:40](../reil/learners/ppo_learner_parallel.py#L40)) — shared trunk + parallel actor heads.
- `shared_layer_sizes: tuple[int, ...]`.
- `actor_layer_sizes: tuple[tuple[int, ...], ...]` (one tuple per head).
- `critic_layer_sizes: tuple[int, ...]`.

**`PPOLearner`** ([ppo_learner.py:739](../reil/learners/ppo_learner.py#L739)) — wraps any of the three PPO models. Constructor: `(model: PPOModel, **kwargs)`. `learn(X, Y) -> dict` returns per-step loss/entropy/KL metrics.

### 6.4 Actor-Critic (A2C) ([reil/learners/actor_critic_learner.py](../reil/learners/actor_critic_learner.py))

- **`DeepA2CModel`** (Keras model). Constructor: `(output_lengths: tuple[int, ...], learning_rate, shared_layer_sizes, actor_layer_sizes=(), critic_layer_sizes=(), critic_loss_coef=1.0, entropy_loss_coef=0.0)`. `call(...)` returns `(action_probs: list[Tensor], values: list[Tensor])`.
- **`A2CLearner`** wraps `DeepA2CModel`.
- `DeepA2CActionProximityModel` — variant referenced in `warfarin_dosing/configs/agents.yaml` `a2cap_agent` with `effect_widths`/`effect_decay_factors`. (Existence confirmed via the config; the class is exposed in the same module.)

### 6.5 Tabular ([reil/learners/lookup_table.py](../reil/learners/lookup_table.py))

- **`TableEntry`** dataclass: `value: T`, `N: int = 0`.
- **`LookupTable`** = `dict[Any, TableEntry[T]]`.
- **`QLookupTable`** (Learner): `(learning_rate, initial_estimate=0.0, minimum_visits=0)`. `predict()` returns `initial_estimate` if visit count < `minimum_visits`; `learn(X, Y)` performs a Robbins-Monro-style update.

### 6.6 Learning-rate schedulers ([reil/learners/learning_rate_schedulers.py](../reil/learners/learning_rate_schedulers.py))

- `LearningRateScheduler(initial_lr, new_rate_function: Callable[[int, float], float])`.
- `ConstantLearningRate(initial_lr)` — `new_rate_function` is constant.
- TensorFlow schedules (`ExponentialDecay`, etc.) are accepted directly because YAML config can instantiate them as objects with their own `__call__`.

---

## 7. Environments (`reil/environments/`)

### 7.1 `Environment` ([reil/environments/environment.py:45](../reil/environments/environment.py#L45))

Base orchestrator. Inherits from `Stateful`.

**Constructor:**
```python
Environment(
    entity_dict: dict[str, BaseAgent | Subject | EntityGenType],
    demon_dict: dict[str, AgentDemon | SubjectDemon] | None,
    interaction_plans: dict[str, Any],
    stopping_criteria: StoppingCriteria | None = None,
    **kwargs,
)
```
- Auto-splits `entity_dict` into `_agents`, `_subjects`, `_agent_demons`, `_subject_demons`, `_instance_generators`. String values pointing at an `InstanceGenerator` are looked up by name.
- `_iterations: defaultdict[str, int]` tracks per-subject iteration count.
- `_active_plan: Plan` — current interaction plan.

**Key methods:**
- `add_entities(entity_dict)`, `add_demons(demon_dict)`, `add_plans(interaction_plans)`.
- `activate_plan(plan_name) -> None`.
- `simulate_pass() -> None` — runs all protocols in the active plan once.
- `report_statistics(unstack=True, reset_history=True) -> dict[tuple[str, str], pd.DataFrame]` — aggregates per-entity statistics.

### 7.2 `Single` ([reil/environments/single.py:42](../reil/environments/single.py#L42))

Plan = single `InteractionProtocol`.
- `simulate_one_pass()` runs the protocol for `n` iterations (`unit ∈ {'iteration', 'epoch', ...}`).
- `remove_entity(entity_names)` — raises `RuntimeError` if an entity is bound to the active plan.
- On `feature/chapters` `report_statistics` was refactored to share an `aggregate_and_transform()` helper with `Sequential` and to handle `None` aggregates gracefully.

### 7.3 `Sequential` ([reil/environments/sequential.py:42](../reil/environments/sequential.py#L42))

Plan = `tuple[InteractionProtocol, ...]`. `simulate_one_pass()` walks the tuple in order, allowing multi-agent / multi-subject sessions in a fixed sequence. Same `aggregate_and_transform()` refactor on `feature/chapters`.

### 7.4 `Task` ([reil/environments/task.py:12](../reil/environments/task.py#L12))

Represents a single training/evaluation step.

**Constructor:**
```python
Task(
    name: str,
    path: pathlib.PurePath | str,
    agent_training_triggers: dict[str, Literal['none','termination','state','action','reward']],
    plan_name: str,
    start_iteration: int = 0,
    max_iterations: int = 1,
    writer: OutputWriter | None = None,
    save_iterations: bool = True,
)
```

- `run_file(environment_filename, path, iteration) -> Single` — loads a pickled `Single`, then calls `run_env`.
- `run_env(env, iteration)` — activates the plan, sets training triggers, calls `simulate_pass`, writes stats.
- `trajectory(env, ...)` — variant that records trajectories.

### 7.5 `Session` ([reil/environments/session.py:18](../reil/environments/session.py#L18))

Top-level experiment driver consumed by `warfarin_main.py`.

**Constructor:**
```python
Session(
    name, path,
    main_task: Task,
    agents: dict[str, Agent | str],
    subjects: dict[str, Subject | str],
    plans,
    demons=None,
    tasks_before=None,
    tasks_after=None,
    tasks_before_iteration=None,
    tasks_after_iteration=None,
    separate_process=None,
    process_type=None,
)
```

- `run()` executes `tasks_before` once, the `main_task` for `max_iterations` (with `tasks_before/after_iteration` per iteration), and `tasks_after` at the end.
- `_run_tasks(...)` (static) supports optional `multiprocessing.Process` spawning per task list — this is what `separate_process: [tasks_after_iteration]` toggles in `warfarin_dosing/configs/sessions.yaml`.

### 7.6 `SessionBuilder` ([reil/environments/session_builder.py:11](../reil/environments/session_builder.py#L11))

YAML-to-`Session` / `Task` factory.

**Constructor:** `SessionBuilder(config_filenames: dict[str, str], config_path: PurePath | str | None, vars_dict: dict[str, str] | None)`.
- `vars_dict` provides the `$variable$` substitutions (e.g., `$h$`, `$action$`, `$project_path$`, `$arm$`, `$sdemon$`).
- `create_session(session_name, parent_session_path='.') -> Session`.
- `create_task(task_name, parent_session_path='.') -> Task`.

Used by `warfarin_main.py` (and `warfarin_single.py` for the `--task=init` branch).

### 7.7 `Trajectory` ([reil/environments/trajectory.py:16](../reil/environments/trajectory.py#L16))

Helper to replay a saved environment with a trajectory dumper attached.

---

## 8. Subjects (`reil/subjects/` — non-healthcare)

### 8.1 `Subject` ([reil/subjects/subject.py:19](../reil/subjects/subject.py#L19))

Inherits from `Stateful`. Constructor: `Subject(sequential_interaction: bool = True, **kwargs)`.

Properties:
- `state: State` — feature definitions per `_extract_sub_components`.
- `reward: Reward` — disabled until `take_effect()` enables it.
- `possible_actions: ActionSet` — action-space generator.

Key methods:
- `is_terminated(_id=None) -> bool` — abstract.
- `take_effect(action, _id=0) -> FeatureSet` — public entry; enables reward, dispatches to `_take_effect`, returns the actually-taken action.
- `_take_effect(action, _id) -> FeatureSet` — abstract.

### 8.2 `SubjectDemon` ([reil/subjects/subject_demon.py:38](../reil/subjects/subject_demon.py#L38))

Wraps a `Subject` and optionally rewrites state or actions.

**Constructor:** `SubjectDemon(subject=None, action_modifier: Modifier | None = None, state_modifier: Modifier | None = None, **kwargs)`.

`Modifier` is a dataclass: `name, cond_state_def, condition_fn(state)->bool, modifier_fn(T)->T`. The condition is evaluated against `cond_state_def` (typically `'day'` in the warfarin demons).

`__call__(subject)` attaches a subject and mirrors `reward`, `statistic`, `is_terminated`, `take_effect`, `reset`. `state(...)` and `possible_actions(...)` delegate to the subject, then run their respective modifier if the condition holds.

### 8.3 Sample domains

| Class | File | Purpose |
| --- | --- | --- |
| `FrozenLake` | `frozen_lake.py:24` | 2D grid: `S`/`F`/`H`/`G`. Actions Up/Down/Left/Right. Reward -1 (hole), +1 (goal). |
| `FrozenRiver` | `frozen_river.py:23` | 1D variant. Actions move 1 or 2 steps left/right. |
| `MNKGame` | `mnkgame.py:19` | Generalized m-by-n-by-k game; constructor `(m, n, k, players)`. |
| `TicTacToe` | `tic_tac_toe.py:14` | `MNKGame(3, 3, 3, players=2)`. |

These are non-healthcare smoke-test domains; the dissertation experiments do not depend on them.

---

## 9. Utilities (`reil/utils/`)

### 9.1 Exploration strategies ([reil/utils/exploration_strategies.py](../reil/utils/exploration_strategies.py))

Protocol surface: `explore(iteration: int) -> bool`.

- `NoExploration` — always `False`.
- `ConstantEpsilonGreedy(epsilon: float)` — `Bernoulli(epsilon)` each call.
- `VariableEpsilonGreedy(epsilon: Callable[[int], float])` — `Bernoulli(epsilon(iteration))`. YAML examples use `"lambda n: 1/(1+n)"`.

### 9.2 Action generation ([reil/utils/action_generator.py](../reil/utils/action_generator.py))

- `CategoricalComponent(name, possible_values, categories, feature_generator)` and `NumericalComponent(name, possible_values, lower, upper, feature_generator)` describe one action coordinate.
- `ActionGenerator(components=None)` — produces FeatureSet actions via Cartesian product of components.

### 9.3 Action distribution modifiers ([reil/utils/action_dist_modifier.py](../reil/utils/action_dist_modifier.py))

- `ScaleFn` base — callable with internal counter; subclasses include:
  - `Constant(value)`.
  - `N_over_N_plus_n(N)` — returns `N / (N + n)`; useful for diminishing exploration multipliers.
  - `Sigmoid(steepness, endpoint)` — `1 / (1 + exp(-steepness*(n - endpoint/2)))`.
- `ActionModifier` (protocol) and `RickerWaveletActionModifier` — used in `run_PPO.py` to weight neighboring discrete actions.

### 9.4 YAML config tooling ([reil/utils/yaml_tools.py](../reil/utils/yaml_tools.py))

- `from_yaml_file(node_reference, filename, path)` — load and parse.
- `parse_yaml(data)` — recursive deserialization. Supports `eval:` keys, `"lambda ..."` strings, nested dicts/lists, fully-qualified class names (looked up via `importlib`), and TF schedule objects.
- `create_component_from_yaml(name, args)` — dynamically import and instantiate a class.

### 9.5 Stopping criteria ([reil/utils/stopping_criteria.py](../reil/utils/stopping_criteria.py))

`StoppingCriteria(monitor: str, mode: Literal['min','max']='min', average_every: int=1, min_delta: float=0., patience: int=0, warm_up: int=0)`.
- `__call__(logs, weights_fn=None) -> bool` returns `True` to stop.
- `get_best() -> (best_value, weights) | None`.

Stops when the monitored metric has not improved by `min_delta` for `patience` calls **after** `warm_up` warm-up calls.

### 9.6 Metrics ([reil/utils/metrics.py](../reil/utils/metrics.py))

- `MetricProtocol` — matches the Keras metric API (`update_state`, `result`, `reset_states`).
- `PTTRMetric(name, mode: Literal['scalar','histogram']='histogram')` — Percent Time in Therapeutic Range (the headline outcome for warfarin).
- `INRMetric` — INR tracking.
- `ActionMetric(name, head_index)` — captures action indices per head (used for dose / duration heads).

### 9.7 Output writer ([reil/utils/output_writer.py](../reil/utils/output_writer.py))

`OutputWriter(filename, path='.', columns=None)` with retrying `write_stats_output(stats_output: dict)` (up to 5 attempts on `PermissionError` — relevant for shared filesystems on cluster runs).

### 9.8 Argument parser ([reil/utils/argument_parser.py](../reil/utils/argument_parser.py))

- `CommandlineArgument(name, type, default, const=None, nargs=None)` dataclass.
- `CommandlineParser(cmd_args, extra_args=...)` — yields `parsed_args: dict[str, Any]`.
- `ConfigParser` — loads multiple YAMLs and supports nested `extract('sessions', name)` style access (used in `warfarin_single.py`).
- `str_to_tuple(s, _type=float, error='raise')` helper.

### 9.9 Instance generators

| Class | File | Purpose |
| --- | --- | --- |
| `InstanceGenerator[T]` | `instance_generator.py:21` | Wraps a single object; emits checkpointed instances at `instance_counter_stops` boundaries. Args include `obj`, `instance_counter_stops`, `auto_rewind`, `save_instances`, `overwrite_instances`, `use_existing_instances`, `save_path`, `filename_pattern`, `state_dumper`. |
| `InstanceGeneratorBatch[T]` | `instance_generator_batch.py:28` | Adds `instance_name_pattern` and `pre_generate_all`. Used by `warfarin_dosing/configs/subjects.yaml` (`training_config`, `validation_config`, etc.). |
| `InstanceGeneratorV2[T]` | `instance_generator_v2.py:22` | Class-based factory: `(cls: type[T], args_generator: Callable | Iterator, is_finite=False, instance_name_pattern='{n:04}')`. Used by `run_PPO.py` / `run_Q.py`. |
| `SubjectGenerator` | (re-exported from `__init__`) | Convenience iterator with `start`/`stops`/`step`. |

### 9.10 Reward shaping ([reil/utils/reil_functions.py](../reil/utils/reil_functions.py))

- `ReilFunction(name, y_var_name, x_var_name=None, length=-1, multiplier=1.0, constant=0.0, interpolate=True)` — parameterized reward function with `__call__(args: FeatureSet) -> float`.
- `CompoundReilFunction` — weighted sum of `ReilFunction`s.
- `NormalizedSquareDistance(center, band_width, amplifying_factor, exclude_first, average)` — the `sq_dist`/`sq_dist_exact`/`sq_dist_modified` rewards used in warfarin experiments.

### 9.11 TensorFlow helpers ([reil/utils/tf_utils.py](../reil/utils/tf_utils.py))

- `set_tf_flags(eager_execution, jit_compile)` — toggles module-level `EAGER_EXECUTION` / `JIT_COMPILE`.
- `entropy(logits)` / `logprobs(logits, indices, index_count)` — `@tf.function` helpers.
- `SerializeTF` — Keras-model save/load roundtrip via temp dirs.
- `TF2UtilsMixin` (`tf_utils.py:139`) — mixin used by PPO/A2C models. Provides:
  - `convert_to_tensor(data: tuple[FeatureSet, ...])`.
  - `mlp_layers(layer_sizes, activation, layer_name_format)`, `mlp_functional(input_, layer_sizes, activation)`, `mlp_functional_w_concat(..., action_per_head, backprop_mode, normalize_before_concat)`.
  - Attributes `_models`, `_callbacks`, `_learning_rate`, `_tensorboard_path`.
- `SummaryWriter` — wraps `tf.summary.create_file_writer` and supports per-key data-type metadata.
- On `feature/chapters` this file has uncommitted modifications (typing / API tweaks; see §13).

### 9.12 Misc helpers

- `reil/utils/functions.py` — `random_choice(f)`, `random_uniform(f)`, `random_normal(f)`, `random_lognormal(f)` — sample from a `FeatureGenerator`. Also `generate_modifier(modifier_fn, condition_fn=None)` used by `wd_utils/experiment.py`.
- `reil/utils/mnkboard.py` — MNK board helpers.

---

## 10. Healthcare (`reil/healthcare/`)

The clinically-motivated subsystem; everything paper-relevant lives here.

### 10.1 Mathematical models (`mathematical_models/`)

**`HealthMathModel`** (base, [health_math_model.py:15](../reil/healthcare/mathematical_models/health_math_model.py#L15)):
- Class attribute `_parameter_generators: FeatureGeneratorSet`.
- `setup(rnd_generators, input_features=None)`, `run(**inputs)`, `generate(rnd_generators, input_features=None, **kwargs)`, `purturb(**kwargs)`.

**`HambergPKPD`** ([hamberg_pkpd.py:27](../reil/healthcare/mathematical_models/hamberg_pkpd.py#L27)) — the canonical 2-compartment model used in dissertation experiments.
- Constructor: `HambergPKPD(randomized: bool = True, cache_size: int = 30)`.
- Models S-warfarin concentration via two-compartment PK + a 9-element vitamin-K cascade state (`A[0..8]`) for PD.
- Hardcoded constants (from Hamberg et al. 2007):
  - `_CL_s_1_1 = 0.314` l/h (oral clearance for CYP2C9 \*1/\*1, 71-year reference).
  - `_CL_s_age = 0.0091` (≈ 9% per decade).
  - `_CL_s_genotypes` — per-CYP2C9-genotype multipliers (`*1/*1`..`*3/*3`).
  - `_V1 = 13.8` l, `_V2 = 6.59` l (central / peripheral volumes).
  - `_k_aS = 2.0` /hr, `_Q = 0.131` l/h.
  - `_E_max = 1.0`, `_gamma = 0.424`.
  - `_EC_50_GG / _EC_50_GA / _EC_50_AA = 4.61 / 3.02 / 2.20` mg/l.
  - `_MTT_1 = 11.6` h, `_MTT_2 = 120` h.
  - `_lambda = 3.61`, `_INR_max = 20.0`, `_baseINR = 1.0`.
- `prescribe(dose: dict[int, float])` and `INR(measurement_days)` are the externally-called methods.

**`HambergPKPD2010`** ([hamberg_pkpd_2010.py:22](../reil/healthcare/mathematical_models/hamberg_pkpd_2010.py#L22)) — alternative one-compartment model from Hamberg et al. 2010, dataset C. Used as a generalization test subject in `run_PPO.py` (`subject_test_10`).
- Constants: `_CL_alleles = {'*1': 0.174, '*2': 0.0879, '*3': 0.0422}` l/h, `_CL_age = -0.00571`, `_V = 14.3` l, `_k_a = 2.0` /hr, `_E_max = 1.0`, `_gamma = 1.15`, `_EC_50_G = 2.05`, `_EC_50_A = 0.96`, `_MTT_1 = 28.6` h, `_MTT_2 = 118.3` h.

**`old_hamberg.HambergPKPD`** — legacy hourly variant retained for backward compat; same constants but lazy hourly evaluation.

**`hamberg_pkpd_tf.HambergPKPD`** — partial TensorFlow rewrite (`@tf.function`-wrapped static methods). Not currently driving experiments per the configs in `warfarin_dosing/configs/`.

### 10.2 Patient classes

**`Patient`** ([patient.py:19](../reil/healthcare/patient.py#L19)) — base class.
- Constructor: `Patient(model: HealthMathModel, random_seed: int | None = None, **feature_values)`.
- Attributes: `_model`, `_rnd_generators`, `feature_gen_set: FeatureGeneratorSet` (set by subclasses), `feature_set: FeatureSet`.
- `generate()` resamples features and reinitializes the model.
- `model(**inputs) -> dict` delegates to `self._model.run(...)`.

**`PatientWarfarinRavvaz`** ([patient_warfarin_ravvaz.py:35](../reil/healthcare/patient_warfarin_ravvaz.py#L35)).
- Constructor: `(model, random_seed=None, randomized=True, allow_missing_genotypes=True, **feature_values)`.
- Feature generators (per the Aurora cohort distribution, Ravvaz et al.):
  - `age` ~ N(67.3, 13.4²), truncated [18, 150] years.
  - `weight` ~ N(199.24, 54.71²) lb, truncated [70, 500].
  - `height` ~ N(66.78, 4.31²) in, truncated [45, 85].
  - `gender`: {F: 0.5314, M: 0.4686}.
  - `race`: {White: 0.9522, Black: 0.0419, Asian: 0.0040, Am. Indian: 0.0018, Pacific Islander: 1e-4}.
  - `tobaco`: {No: 0.9067, Yes: 0.0933}.
  - `amiodarone`: {No: 0.8849, Yes: 0.1151}.
  - `fluvastatin`: {No: 0.9998, Yes: 0.0002}.
  - `CYP2C9`: {`*1/*1`: 0.6739, `*1/*2`: 0.1486, `*1/*3`: 0.0925, `*2/*2`: 0.0651, `*2/*3`: 0.0197, `*3/*3`: 2e-4}.
  - `VKORC1`: {G/G: 0.3837, G/A: 0.4418, A/A: 0.1745}.
  - `sensitivity` — derived from `(CYP2C9, VKORC1)` combination in `_generate_sensitivity()` → `{normal, sensitive, highly_sensitive}`.
  - Plus model-side: `MTT_1`, `MTT_2`, `V1`, `V2`, `EC_50`, `CL_S_cyp_1_1`.
- Variants in the same module:
  - `PatientWarfarinBalanced` — uniform genotype prior.
  - `PatientWarfarinOversampled` — uniform CYP2C9 prior, Ravvaz VKORC1.

### 10.3 Subjects

**`HealthSubject`** ([subjects/health_subject.py:19](../reil/healthcare/subjects/health_subject.py#L19)) — base class for medical subjects.
- Constructor: `(patient, measurement_name, measurement_range, max_day, duration_range, backfill, duration_step=None, duration_values=None, default_duration=None, **kwargs)`.
- Auto-generates feature definitions: `{measurement_name}_history`, `daily_{measurement_name}_history`, `day`, `duration_history`, `duration`.
- Per-day state (`_full_measurement_history`) vs decision-point state (`_decision_points_measurement_history`).
- `_take_effect(action, subject_id) -> FeatureSet` applies action, runs the patient model, updates histories, decides termination.

**`DosingSubject`** ([subjects/dosing_subject.py:17](../reil/healthcare/subjects/dosing_subject.py#L17)).
- Constructor adds `(dose_range, decision_mode, dose_step=None, decision_values=None, decision_range=None, round_to_step=True, ...)`.
- `decision_mode` ∈ {
  `'dose'`, `'dose_change'`, `'dose_percent_change'`,
  `'dose_duration'`, `'dose_change_duration'`, `'dose_percent_change_duration'`,
  plus `*_joint` variants tying dose & duration into a single multi-head action }.
- Action masking supported via the underlying `FeatureGenerator`.

**`Warfarin`** ([subjects/warfarin.py:19](../reil/healthcare/subjects/warfarin.py#L19)) — the canonical subject used in every dissertation experiment.
- Constructor: `(patient, INR_range, dose_range, dose_step, duration_range, duration_step, max_day, backfill=True, **kwargs)`.
- Pre-built **state definitions** (referenced by name in `interaction_protocols.yaml`):
  - `'age'`, `'patient_basic'` (age + CYP2C9 + VKORC1), `'patient_w_sensitivity'`, `'patient_w_dosing'`, `'patient_w_dosing_*i*'` for short histories (`patient_w_dosing_01`, `..._02`, `..._03`), `'patient_w_full_dosing'`, `'patient_for_baseline'`.
- Pre-built **action definitions**:
  - Discrete dose sets: `'237_15'`, `'daily_15'`, `'free_15'`, `'semi_15'`, `'weekly_15'`, plus `'237_05'`, `'237_10'`, `'semi_05'`, `'semi_10'`.
  - Joint dose+duration: `'delta'`, `'percent'`, `'percent_semi'`, `'percent_semi_joint'`, `'semi'`.
- Pre-built **rewards**: `'sq_dist'`, `'sq_dist_exact'`, `'sq_dist_modified'`, `'PTTR_exact'`, `'dose_change'`, `'dist'`, `'custom_distance'` variants, plus `'no_reward'`.
- Termination: INR outside `[0.5, 10]` or `day >= max_day`.

**`WarfarinIncrementalAction`** ([subjects/warfarin_incremental_action.py:27](../reil/healthcare/subjects/warfarin_incremental_action.py#L27)) — `Warfarin` subclass with a refinable discrete action set (additive or multiplicative). Used by `run_incremental_action.py`.

### 10.4 Baseline protocol agent

**`WarfarinAgent`** ([agents/warfarin_agent.py:20](../reil/healthcare/agents/warfarin_agent.py#L20)).
- Constructor: `(study_arm: str | ThreePhaseDosingProtocol = 'aaa', dose_range=(0, 15), duration_range=(1, 28), **kwargs)`.
- Dispatches to a `ThreePhaseDosingProtocol` named by `study_arm`. The five canonical arms are wired in `dosing_protocols/warfarin/ravvaz_dosing_protocols.py`:

| Arm | Initiation (days 1–~2) | Adjustment | Maintenance |
| --- | --- | --- | --- |
| **AAA** | Aurora | Aurora | Aurora |
| **CAA** | IWPC clinical (days 1–2) | Aurora | Aurora |
| **PGAA** | IWPC pharmacogenetic | Aurora | Aurora |
| **PGPGA** | IWPC modified (days 1–3) | Lenzini (days 4–5) | Aurora |
| **PGPGI** | IWPC modified (days 1–3) | Lenzini (days 4–5) | Intermountain |

`act(state, subject_id, actions, iteration)` reconstructs the patient dict, calls `_protocol.prescribe(patient_dict)`, then re-encodes the result into the action FeatureSet.

### 10.5 PPO agents for warfarin

**File:** [reil/healthcare/agents/warfarin_ppo_agent.py](../reil/healthcare/agents/warfarin_ppo_agent.py) — modified on `feature/chapters`.

Five classes implemented:

1. **`PPO4WarfarinAgent`** (line 19) — base PPO agent for the warfarin domain.
   - Adds metrics: `PTTR_h`/`INR_h` (histograms), `PTTR`/`INR` (scalars), `dose` (`ActionMetric` index 0), `duration` (`ActionMetric` index 1), and one `modifier_<name>` mean metric per `ActionModifier`.
   - Tracks `_previous_action` (zero-initialised tensors of `action_per_head` length) for momentum support.
   - Constructor adds: `momentum_coef: float = 0.0`, `momentum_mode: Literal['most recent','carry'] | None = None`, `action_modifiers: list[ActionModifier] | None = None`.
   - `act(...)` predicts logits, blends previous-action momentum, applies each modifier in `_action_modifiers`, applies action masking, samples (or argmaxes), then returns the resulting FeatureSet.

2. **`PPO4Warfarin2PhaseAgent`** (line 152) — sub-classes `PPO4WarfarinAgent`. Adds initiation-phase delegation.
   - Constructor adds: `init_agent: BaseAgent` (e.g., a `RandomAgent` or `WarfarinAgent('pgaa')`), `switch_day: int`, `init_state_comps: tuple[str, ...]`, `main_state_comps: tuple[str, ...]`.
   - `act(...)`: if `state['day'] < switch_day`, restricts state to `init_state_comps` and defers to `init_agent`; otherwise restricts to `main_state_comps` and uses PPO.
   - `_prepare_training(history)` filters out the day-`day` rows so the policy is trained only on post-switch transitions.

3. **`ActionSplitter`** (line 199) — utility wrapping an action generator to expose only one head's actions (used to chain dose-only and duration-only agents).

4. **`PPO4Warfarin2PartAgent`** (line 228) — two coupled PPO agents (`dose_agent`, `duration_agent`) with an optional `training_switch` schedule (e.g., `(('all', 10000),)`) and a `dose_first` flag controlling head order. Splits history into per-head streams via `_split_history`.

5. **`PPO4WarfarinSeparateAgent`** (line 505) and **`PPO4WarfarinSeparateSimpleRewardAgent`** (line 524) — override `_split_history` to apply duration-specific reward shaping:
   - Separate: `reward = -(tau * |INR-2.5| * 0.3)` if out-of-range, else `-(28-tau) * (0.5 - |INR-2.5|)`.
   - SimpleReward: `reward = -1 if (|INR-2.5| > 0.5 and tau > 1) else (tau - 7) / 28`.

### 10.6 Dosing protocols

`dosing_protocols/dosing_protocol.py`:
- `DosingDecision(dose: float, duration: int | None)` dataclass.
- `DosingProtocol.prescribe(patient, additional_info)` → `(DosingDecision, dict)`; `reset()` clears flags.

`three_phase_dosing_protocol.py:16` — `ThreePhaseDosingProtocol(initial_protocol, adjustment_protocol, maintenance_protocol)` dispatches by `patient['day']`.

Per-protocol files in `dosing_protocols/warfarin/`:

- `aurora.py:15` — Aurora 2017 algorithm: day-1/2 fixed initiation, then lookup-table adjustments driven by recent INR and dose history. Internal state tracks `red_flag`, `skip_dose`, `new_dose`, `number_of_stable_days`. Retest intervals 2, 7, or 28 days.
- `iwpc.py:22` — methods `'pharmacogenetic'`, `'clinical'`, `'modified'`, `'loading_dose'`.
- `gage.py:17` — methods `'pharmacogenetic'`, `'clinical'`.
- `lenzini.py` — Lenzini 2010 adjustment regression.
- `intermountain.py` — Intermountain dosing for maintenance phase; `Intermountain(enforce_day_ge_8=False)`.
- `ravvaz_dosing_protocols.py` — assembles AAA/CAA/PGAA/PGPGA/PGPGI from the above (see table in §10.4).
- `custom_protocols.py` — extension point for user-defined protocols.

### 10.7 Trajectory dumper ([trajectory_dumper.py:12](../reil/healthcare/trajectory_dumper.py#L12))

`TrajectoryDumper(filename, path)` extends `FeatureSetDumper`. `_dump(component, additional_info, filename, path) -> bool`:
- Extracts daily measurement series via regex on keys matching `daily_*_history`.
- Builds a DataFrame with columns `[measurements..., dose, decision_points, day, <patient features...>, <additional_info>]`.
- `decision_points` is a 0/1 marker showing where the agent actually made a choice (one `1` per `duration` interval).
- Drops the last (incomplete) day.
- Appends to CSV; returns `False` (and skips) on `PermissionError`.

### 10.8 `healthcare/__init__.py` re-exports

- `Patient`, `PatientWarfarinRavvaz` (top-level).
- Submodules `agents`, `subjects`, `mathematical_models`, `dosing_protocols`.
- `TrajectoryDumper` from the trajectory dumper file.

`reil/healthcare/agents/__init__.py` exports only `WarfarinAgent`. **The PPO warfarin agents are not auto-imported** — callers must do `from reil.healthcare.agents.warfarin_ppo_agent import PPO4WarfarinAgent` (or `...Agent2Phase`/`...2PartAgent`/`...SeparateAgent`/`...SeparateSimpleRewardAgent`).

---

## 11. Legacy (`reil/legacy/`)

The `legacy` package preserves older implementations so that historical `.pkl` checkpoints continue to load. Includes:

- `feature.py`, `reildata.py` — old datatypes superseded by `datatypes/feature.py`.
- `risk.py`, `snake.py`, `windy_gridworld.py` — sample domains.
- `warfarin_cluster_based_agent.py`, `weka_clustering.py`, `test_warfarin_agent.py` — legacy warfarin work.

Do not import from `legacy` in new code. The `CustomUnPickler` mappings handle the renames automatically.

---

## 12. Top-level package globals (`reil/__init__.py`)

- `FILE_FORMAT: Literal['pbz2', 'pkl'] = 'pkl'` — switch by `set_file_format(...)`.
- Random generator triple stored on module:
  - `RANDOM_GENERATOR: random.Random`.
  - `RANDOM_GENERATOR_NP: np.random.Generator`.
  - `RANDOM_GENERATOR_TF: tf.random.Generator`.
- `random_generator()`, `random_generator_np()`, `random_generator_tf()` accessors.
- `random_generators_from_seed(seed) -> tuple` — convenience triple constructor.
- `set_reil_random_seed(seed)` — sets the module globals; used by `run_PPO.py`, `run_Q.py`, `warfarin_main.py`, `warfarin_single.py` (all pass `1234` or `12345`).
- `random_generator_context(gen=None, gen_np=None, gen_tf=None)` — context manager to scope an alternate generator.

`__version__` is sourced from `_version.py` (versioneer).

---

## 13. `feature/chapters` branch — uncommitted modifications

These files are modified in the working tree of `feature/chapters` but not yet committed. They are summarized here so the master-based reference above can be re-read with the deltas in mind. Use `git diff master -- <file>` for byte-level diffs.

- **`reil/agents/agent.py`** — minor typing and a defensive `hasattr` check for `reset_states()` vs `reset_state()` when iterating metrics.
- **`reil/agents/agent_demon.py`** — `__call__(main_agent)` now mirrors `_summary_writer`, `_learner`, `_computed_metrics`, `_metrics` if present. `learn()` returns `dict[str, float]` (previously `None`).
- **`reil/agents/ppo_agent.py`** — `_update_metrics` accepts list/tuple `y` and uses `tf.expand_dims` on 1-D tensors before invoking `action_rank`.
- **`reil/environments/sequential.py`** and **`reil/environments/single.py`** — shared `aggregate_and_transform()` helper inside `report_statistics()`; null-safe handling of aggregator absence.
- **`reil/healthcare/agents/warfarin_ppo_agent.py`** — extends `PPO4Warfarin2PartAgent` and adds the two `Separate*` reward-shaping variants. (See §10.5 above for the current class list — these classes only exist on `feature/chapters`.)
- **`reil/learners/ppo_learner.py`** — refinements to `train_step` / `train_actor`; addition of `PPONeighborEffect`.
- **`reil/learners/ppo_learner_tandem.py`** — staged-training plumbing (`training_switch`, `_freeze_layers`, `backprop_mode`).
- **`reil/utils/tf_utils.py`** — minor API touch-ups; `mlp_functional_w_concat` exists here.

These deltas are referenced by `warfarin_dosing/run_PPO.py` and `run_incremental_action.py`, which expect the warfarin PPO chain to be importable.

---

## 14. Notes on possible discrepancies

These are surface-level observations spotted while compiling this reference. They are **not** a substitute for the thorough dissertation-vs-implementation comparison the user plans to run next; they only flag things worth verifying.

1. **PPO module/class names referenced by `warfarin_dosing/run_PPO.py` and `run_incremental_action.py` are stale.**
   Both files do `from reil.healthcare.agents.ppo_warfarin import PPO4Warfarin2Phase` / `PPO4Warfarin`. The actual module is `reil.healthcare.agents.warfarin_ppo_agent`, with classes `PPO4Warfarin2PhaseAgent` and `PPO4WarfarinAgent` (suffix `Agent`). These imports will fail as-is; either the scripts need updating or an alias module needs to be added.

2. **`run_incremental_action.py` imports `reil.datatypes.feature_set_dumper`,** but `FeatureSetDumper` lives in `reil.datatypes.feature` in the current source. Same stale-import flavour as (1).

3. **`run_PPO.py` passes `default_interval=7` to the warfarin subject** (in `WarfarinGenerator.__next__`), but the codebase has renamed `interval` to `duration` (commit `9b141f2`, "duration instead of interval"). The keyword would now be `default_duration` (also encoded in the `CustomUnPickler` mapping `'interval → duration'` substring rule). Likely the run script predates the rename.

4. **Baseline-arm letter casing.** `WarfarinAgent.__init__` lowercases the `study_arm` string before lookup, and `wd_utils/experiment.py` calls `WarfarinAgent(study_arm=spec['method'].lower())`. The dissertation prose often capitalises the arm names (`AAA`, `CAA`, …); the implementation accepts either. Worth a quick consistency pass when the comparison runs.

5. **A2C config (`warfarin_dosing/configs/agents.yaml` `a2c_agent`)** specifies `output_lengths: [31, 7]` — i.e., 31 dose bins and 7 duration bins. This is a *hard-coded* shape; if a later experiment changes the action discretisation (e.g., to `237_15`), the actor-critic config will not match. The PPO configs are driven through `$action$` substitution and avoid this issue.

6. **Two parallel orchestration paths.**
   - `warfarin_main.py` + `warfarin_single.py` + `wd_utils/experiment.py` use **`SessionBuilder` from YAML** (`configs/sessions.yaml` etc.) and target the cluster job-scheduler (`run.py`).
   - `run_PPO.py`, `run_Q.py`, `run_Q_lookahead.py`, `run_incremental_action.py` **bypass `SessionBuilder`** and build a `Single` environment directly in Python.
   Both paths coexist; the canonical dissertation experiments in `configs/sessions.yaml` go through `SessionBuilder`, but the standalone PPO scripts (which back the chapter-3 work) go through the second path. Verify this matches what each canonical doc actually claims.

7. **`hamberg_pkpd_tf.HambergPKPD`** exists as a parallel TensorFlow rewrite but is not referenced by any config. It may be in-flight or abandoned — flag for the next comparison pass.

8. **`HambergPKPD2010`** is wired in `run_PPO.py` as the *test* PK/PD model (`subject_test_10`) but never as a training subject. If the dissertation claims dual training, that's worth verifying.

---

*End of file. Last reviewed against master HEAD `ba3824bf`. `feature/chapters` deltas summarised in §13.*
