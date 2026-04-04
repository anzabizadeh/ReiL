# ReiL Architecture & Design

This document explains the design decisions, architectural patterns, and rationale behind ReiL. It's intended for developers considering extending, modifying, or contributing to ReiL.

## Design Principles

### 1. Hierarchical Abstraction via Inheritance

**Principle**: Build complex RL systems by composing simpler abstractions.

**Implementation**:
- **ReilBase**: Low-level concerns (naming, logging, persistence)
- **Stateful**: Entity-level concerns (state management, entity registration, statistics)
- **Agent/Subject/Environment**: Domain-specific concerns (learning, environment simulation)

**Rationale**: Separation of concerns allows each layer to be tested, understood, and extended independently. A developer can work with Stateful without understanding the full RL framework.

**Example**: When creating a custom Subject for a new domain:
```python
class MedicalSubject(Subject):  # Inherits Stateful, which inherits ReilBase
    def step(self, action):
        # Medical domain logic
        pass
```

The medical subject automatically inherits logging, persistence, and state management from parent classes.

### 2. Configuration-Driven Instantiation

**Principle**: Decouple configuration from code; allow runtime customization without recompilation.

**Implementation**: YAML configs map fully-qualified class names to constructor parameters. The [yaml_tools.py](../reil/utils/yaml_tools.py) module recursively instantiates objects.

**Rationale**: Researchers often reconfigure experiments (change learning rates, swap agents, adjust buffer sizes) without recompiling Python. YAML is human-readable and version-controllable.

**Example**:
```yaml
# Config
agent:
  reil.agents.DeepQLearning:
    name: my_agent
    learner:
      reil.learners.q_learner.QLearner:
        learning_rate: 1e-3

# Python
config = load_config_dict('config.yaml')  # Deserializes to live objects
agent = config['agent']  # Ready to use
```

### 3. Feature/FeatureSet as Universal Communication

**Principle**: All inter-component communication (state, action, reward) uses the same abstraction.

**Implementation**: [Feature](../reil/datatypes/feature.py) is an immutable dataclass with metadata (type, bounds, categories, normalization). [FeatureSet](../reil/datatypes/feature.py) is a dict-like container of Features.

**Rationale**: 
- **Type Safety**: Categorical vs. numerical features have different constraints enforced at creation time
- **Automatic Normalization**: Numerical features normalize to `[0, 1]` for neural networks
- **Missing Value Handling**: Graceful encoding of absent data (e.g., missing genetic markers)
- **Serialization**: FeatureSets are fully serializable with metadata intact

**Example**:
```python
# Agent observes state
state = FeatureSet({
    'patient_age': Feature.numerical('age', value=65, lower=0, upper=120, normalized=(0.54,)),
    'genotype': Feature.categorical('genotype', value='VKORC1_1639G>A', 
                                    categories=(...), normalized=(1, 0, 0))  # one-hot
})

# Agent computes action
action = FeatureSet({
    'warfarin_dose': Feature.numerical('dose_mg', value=5.0, lower=1.0, upper=10.0, normalized=(0.45,))
})

# Subject computes reward
reward_value = 1.0 if patient_INR_in_therapeutic_range else -0.5
reward = FeatureSet({'reward': Feature.numerical('r', value=reward_value)})
```

### 4. Entity Registration for Relationship Tracking

**Principle**: Agents and Subjects need to know about each other for proper coordination.

**Implementation**: [Stateful](../reil/stateful.py) maintains an `EntityRegister` of related entities. Subjects track agents; agents track subjects.

**Rationale**: Enables features like:
- Statistics computed per entity (e.g., per-agent win rates)
- Targeted state observations (e.g., partial observability per agent)
- Demon experiments (modifying specific agent/subject behaviors)

**Example**:
```python
subject.register(agent_1)  # Subject knows about agent_1
subject.register(agent_2)  # Subject knows about agent_2

stats = subject.statistic(agent_1._id, 'episode_reward')  # Get agent_1-specific stats
```

### 5. Backward Compatibility via Serialization Mapping

**Principle**: Long-lived experiments need to load models trained years ago, despite codebase evolution.

**Implementation**: [CustomUnPickler](../reil/serialization.py) maintains a mapping of old class paths → new class paths.

**Rationale**: Researchers often return to old experiments or need to deploy models in production. Renames/refactors should not break legacy checkpoints.

**Example**:
```python
# OLD: reil.agents.deep_q.DeepQAgent
# NEW: reil.agents.deep_q_learning.DeepQLearning

# In serialization.py:
UNPICKLER_MAPPING = {
    'reil.agents.deep_q.DeepQAgent': 'reil.agents.deep_q_learning.DeepQLearning',
    ...
}

# Now old checkpoints load seamlessly
```

## Configuration Loading Pipeline

Understanding how configs are loaded will help when:
- Debugging "why isn't my config working?"
- Adding new entity types to instantiate from YAML
- Validating parameter passing

### Step-by-Step

1. **Read YAML file** → dict representation:
   ```yaml
   agent:
     reil.agents.DeepQLearning:
       name: my_agent
       discount_factor: 0.95
   ```

2. **Variable substitution** (optional):
   ```python
   config = load_config_dict('agents.yaml', variable_dict={'name': 'exp_1'})
   # $name$ becomes 'exp_1'
   ```

3. **Recursive entity instantiation**:
   - For each top-level key (e.g., 'agent'), find the fully-qualified class name (first nested key)
   - Import the class dynamically via `importlib`
   - Recursively instantiate all nested objects (e.g., 'learner', 'buffer')
   - Call `ClassName(**config_dict)` to construct the object

4. **Return live objects** (not dict):
   ```python
   config['agent']  # Returns a DeepQLearning instance, ready to use
   ```

### Code Reference

- **Loading**: [reil/utils/yaml_tools.py](../reil/utils/yaml_tools.py#L50-L100) — `load_config_dict()` function
- **Instance creation**: [reil/utils/yaml_tools.py](../reil/utils/yaml_tools.py#L120-L150) — `create_object()` function

## State Representation Design

### Why FeatureSet for State?

RL systems need to represent observations in diverse formats:
- Tabular: simple (age, dosage) pairs
- Image-based: pixel arrays
- Mixed: some scalars + sparse categorical data

Rather than create separate systems for each, ReiL uses **Feature** as a universal abstraction:

```python
# Tabular state
state = FeatureSet({
    'age': Feature.numerical(...),
    'dosage': Feature.numerical(...)
})

# Could also include image
state['visual'] = Feature.numerical('pixels', value=np.array([...]))  # 2D array
```

### Normalization Strategy

Neural networks expect inputs in a narrow range (typically `[0, 1]`). FeatureSet automates this:

```python
# Automatic normalization on Feature creation
age_feature = Feature.numerical('age', value=65, lower=0, upper=120)
# Internally: normalized=(0.541,)  # (65 - 0) / (120 - 0)

# For categorical
dose_feature = Feature.categorical('dose', value='high', 
                                  categories=('low', 'med', 'high'))
# Internally: normalized=(0, 0, 1)  # one-hot encoded
```

Learners extract these normalized values and feed them to neural networks.

### Missing Value Encoding

Some real-world data is incomplete (e.g., genetic test not performed). FeatureSet has two strategies:

1. **MISSING Value**:
   ```python
   Feature.categorical('genotype', value=MISSING, categories=(...))
   # Downstream code detects MISSING and handles gracefully
   ```

2. **NoneFeature**:
   ```python
   from reil.datatypes.feature import NoneFeature
   state['genotype'] = NoneFeature('genotype')
   # Explicitly marks feature as absent
   ```

Learner models should handle these patterns (e.g., masking in neural networks).

## Entity Registration & Multi-Entity Experiments

### Problem It Solves

In multi-agent / multi-subject scenarios:

```python
# Two agents, one subject
agent1, agent2, subject = ...
env = Sequential(subjects=[subject], agents=[agent1, agent2])
```

The subject needs to:
- Know which agents are observing it
- Compute separate statistics per agent
- Support partial observability (agent 1 sees full state, agent 2 sees partial)

### Solution: EntityRegister

Each Stateful object maintains an `EntityRegister`:

```python
subject.register(agent1)
subject.register(agent2)

# Now subject can query per-agent stats
stats_for_agent1 = subject.statistic(agent1._id, 'episode_reward')
```

### Code Reference

- [reil/datatypes/entity_register.py](../reil/datatypes/entity_register.py) — EntityRegister implementation
- [reil/stateful.py](../reil/stateful.py#L100-L120) — `register()` / `deregister()` methods

## Sub-Component Extraction

### Pattern

ReiL automatically discovers and manages nested ReiL objects via a naming convention:

```python
class MyAgent(Agent):
    def __init__(self, custom_buffer, custom_exploration):
        super().__init__(...)
        self._sub_comp_buffer = custom_buffer  # Prefix: _sub_comp_
        self._sub_comp_exploration = custom_exploration
```

The Stateful base class calls `_extract_sub_components()` during init, which:
1. Finds all attributes starting with `_sub_comp_`
2. Registers them as sub-components
3. Makes them available for state tracking, serialization, etc.

**Benefit**: Nested objects (learner, buffer, exploration strategy) are automatically tracked and can be checkpointed independently.

## Learning Rate Scheduling

### Design

RL training often benefits from decaying learning rates over time. ReiL supports this via:

1. **Constant learning rate** (simple cases):
   ```yaml
   learning_rate: 1e-3
   ```

2. **TensorFlow decay schedules** (flexible):
   ```yaml
   learning_rate:
     tensorflow.keras.optimizers.schedules.ExponentialDecay:
       initial_learning_rate: 0.01
       decay_steps: 100
       decay_rate: 0.96
   ```

3. **Lambda functions** (custom):
   ```yaml
   epsilon:  # Exploration decay
     "lambda n: 1/(1+n)"
   ```

These are all handled uniformly through the config loading system.

## Serialization & Checkpoint Strategy

### Challenge

Long-running experiments may:
- Crash mid-training (need to resume)
- Train on cluster (need to distribute, checkpoint, collect results)
- Train for months (want intermediate checkpoints)

### Solution

Every ReilBase object can save/load via:

```python
agent.save('checkpoint_ep_100')  # Pickles entire agent (learner, buffer, etc.)
agent.load('checkpoint_ep_100')  # Restores from pickle

# Underlying: [reil/serialization.py](../reil/serialization.py)
```

### Backward Compatibility

If code changes (e.g., class rename), [CustomUnPickler](../reil/serialization.py#L50-L80) remaps old class names:

```python
# Example: Old checkpoint has reil.agents.deep_q.DeepQAgent
# New code renamed to reil.agents.deep_q_learning.DeepQLearning
# CustomUnPickler automatically maps: AlreadyExistError bypassed
```

## Extension Points

### 1. Custom Agent

**What to override**:
- Inherit from `Agent` or `BaseAgent`
- Implement `act(state: FeatureSet) -> FeatureSet`
- Pass custom learner to constructor

**Example**:
```python
class MyAgent(Agent):
    # act() inherited from BaseAgent, uses learner internally
    pass

agent = MyAgent(learner=MyLearner(...), exploration_strategy=...)
```

### 2. Custom Subject

**What to override**:
- Inherit from `Subject`
- Define state in `__init__` via `self.state.add_definition()`
- Implement `step(action: FeatureSet) -> (FeatureSet, float)`

**Example**:
```python
class MySubject(Subject):
    def __init__(self):
        super().__init__()
        self.state.add_definition('my_feature', Feature.numerical(...))
    
    def step(self, action):
        # Update internal state, compute reward
        new_state = FeatureSet({...})
        reward = ...
        return new_state, reward
```

### 3. Custom Learner

**What to override**:
- Inherit from `Learner`
- Implement `learn(inputs, targets) -> None`
- Optionally override `save_checkpoint()` / `load_checkpoint()`

**Example**:
```python
class MyLearner(Learner):
    def learn(self, inputs, targets):
        # Training logic here
        pass
```

### 4. Custom Exploration Strategy

**What to override**:
- Inherit from `ExplorationStrategy`
- Implement `should_explore(state, iteration) -> bool`

**Example**:
```python
class MyExploration(ExplorationStrategy):
    def should_explore(self, state, iteration):
        # Custom decide whether to explore
        return iteration < 1000
```

### 5. Custom Environment

**What to override**:
- Inherit from base Environment
- Implement orchestration logic (when to step subjects, when to trigger learning)

**Example**: See [reil/environments/sequential.py](../reil/environments/sequential.py), [reil/environments/single.py](../reil/environments/single.py)

## Performance Considerations

### Vectorization

For large-scale experiments:
- Use batch processing in learners (vectorize neural network updates)
- Use circular buffers with efficient roll operations
- Consider distributed training via `compute_resources` integration (see [warfarin_dosing/run.py](../../warfarin_dosing/run.py))

### Memory

- **Buffer size**: Trade-off between stability (large buffer) and memory/GPU VRAM (small buffer)
- **Checkpoint frequency**: Save rarely to avoid I/O overhead, but frequently enough to avoid catastrophic loss

### Debugging

- Enable logging to diagnose training issues:
  ```python
  agent = Agent(..., logger_level=logging.DEBUG)
  ```
- Dump trajectories to disk for offline analysis
- Use TensorBoard integration for learning curves:
  ```yaml
  tensorboard_path: ./logs
  tensorboard_filename: my_experiment
  ```

## Known Limitations & Future Work

### Current Limitations

1. **No distributed agents**: Agents train on single machine (though [warfarin_dosing/run.py](../../warfarin_dosing/run.py) supports multi-process job submission)
2. **No graph-based state**: States must be FeatureSets (not graph structures)
3. **Limited exploration strategies**: Only epsilon-greedy variants; no Upper Confidence Bound (UCB) etc.

### Potential Extensions

- Multi-GPU training via `tf.distribute.Strategy`
- Graph neural networks for relational states
- Hierarchical RL (options framework)
- Inverse RL / imitation learning

## Recommended Reading

1. Start with [../reil/.instructions.md](../reil/.instructions.md) for code-level details
2. See [../copilot-instructions.md](../copilot-instructions.md) for high-level guidance
3. Explore tests in [../tests/](../tests/) for usage examples
4. Examine [../../warfarin_dosing/](../../warfarin_dosing/) for a real-world application

## Quick Reference: Key Files

| Concern | File |
|---------|------|
| Base class | [../reil/reilbase.py](../reil/reilbase.py) |
| Entity management | [../reil/stateful.py](../reil/stateful.py) |
| Features & state | [../reil/datatypes/feature.py](../reil/datatypes/feature.py) |
| Config loading | [../reil/utils/yaml_tools.py](../reil/utils/yaml_tools.py) |
| Serialization | [../reil/serialization.py](../reil/serialization.py) |
| Agents | [../reil/agents/agent.py](../reil/agents/agent.py) |
| Subjects | [../reil/subjects/subject.py](../reil/subjects/subject.py) |
| Environments | [../reil/environments/](../reil/environments/) |
| Learners | [../reil/learners/](../reil/learners/) |
