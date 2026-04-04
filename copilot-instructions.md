# ReiL & Warfarin_Dosing: Shared AI Instructions for RL Projects

This document provides high-level guidance for AI assistants working on reinforcement learning projects using ReiL (or projects that apply ReiL to domain problems like warfarin dosing).

## When to Use This Document

- You're designing or modifying an agent, subject, or environment
- You're configuring training experiments via YAML
- You're troubleshooting issues with state representation, learning convergence, or entity interactions
- You're extending ReiL with custom learners, exploration strategies, or domain-specific subjects

For deeper architectural details, see **[.instructions.md](.instructions.md)** in ReiL or project-specific `.instructions.md` in warfarin_dosing.

## RL Abstractions: Quick Decision Guide

### Choosing an Agent Type

**Question**: What does the agent need to do?

| Agent Type | Use When | Config Example |
|-----------|----------|-----------------|
| **Deep Q-Learning** | Discrete actions, value-based learning | See [warfarin_dosing/configs/agents.yaml](warfarin_dosing/configs/agents.yaml) `dqn_agent_setting` |
| **PPO** | Continuous or discrete actions, policy gradient, stable training | `reil.agents.PPO` with `PPOLearner` |
| **Actor-Critic (A2C)** | Policy + value function, lower variance gradients | See `a2c_agent` in agents.yaml |
| **Random** | Baselines, exploration, testing | `reil.agents.RandomAgent` |
| **User Agent** | Player-controlled actions (e.g., human trials) | `reil.agents.UserAgent` |
| **Warfarin Protocol** | Baseline dosing strategies (clinical protocols) | `reil.healthcare.agents.WarfarinAgent` with `study_arm` (AAA, CAA, PGAA, PGPGI, PGPGA) |

**Decision Flow**:
1. Is the problem primarily about comparing against clinical baselines? → Use WarfarinAgent with different study_arm values
2. Do you need a learnable agent? → Pick DQN (discrete) or PPO (continuous/flexible)
3. Do you need low-variance policy gradient learning? → Use A2C (Actor-Critic)
4. Do you just need random behavior? → Use RandomAgent

### Choosing a Subject Type

**Question**: What entity does the agent interact with?

| Subject Type | Use When | Location |
|-------------|----------|----------|
| **Warfarin** | Modeling patient warfarin dosing (PK/PD) | [reil/healthcare/subjects/warfarin.py](reil/healthcare/subjects/warfarin.py) |
| **Custom Subject** | Domain-specific simulation (e.g., game rules) | Inherit from `Subject`, implement `step()` |

**For Warfarin Subject**:
- Define patient via `PatientWarfarinRavvaz` + `HambergPKPD` or `HambergPKPD2010` model
- Customize state features in subject config (see [warfarin_dosing/configs/subjects.yaml](warfarin_dosing/configs/subjects.yaml))
- Use `TrajectoryDumper` to record (state, action, reward) sequences for analysis

### Choosing a Learner

**Question**: What learning algorithm should the agent use?

| Learner | Best For | Key Parameters |
|---------|----------|-----------------|
| **QLearner** (Deep Q) | Discrete action spaces, value-based control | `learning_rate`, `hidden_layer_sizes`, `buffer_size`, `batch_size` |
| **PPOLearner** | Policy gradient, both discrete/continuous | `learning_rate`, `shared_layer_sizes`, `batch_size`, `epochs` |
| **A2CLearner** | Actor-Critic (lower variance) | `learning_rate`, `shared_layer_sizes`, `entropy_loss_coef` |

**Typical Workflow**:
1. Instantiate learner with a model (DeepQModel, DeepPPOModel, etc.)
2. Pass learner to agent constructor
3. During training, agent calls `learner.learn(inputs, targets)` based on `training_trigger`

## State Representation Patterns

### Defining State (In a Subject)

```python
from reil.datatypes.feature import Feature, FeatureSet

class MySubject(Subject):
    def __init__(self, ...):
        super().__init__(...)
        
        # Add state definition: what features constitute the state?
        self.state.add_definition(
            'my_scalar_feature',
            Feature.numerical('value', lower=0, upper=100)
        )
        
        self.state.add_definition(
            'my_category_feature',
            Feature.categorical('dosing_category', categories=('low', 'med', 'high'))
        )
    
    def step(self, action: FeatureSet) -> tuple[FeatureSet, float]:
        # Update internal state based on action
        new_state = FeatureSet()
        new_state['value'] = Feature.numerical('value', value=50, lower=0, upper=100)
        new_state['dosing_category'] = Feature.categorical('dosing_category', value='med', categories=('low', 'med', 'high'))
        
        reward = self._compute_reward(action)
        return new_state, reward
```

**Key Points**:
- Always define `lower`/`upper` for numerical features (ensures normalization)
- Always define `categories` for categorical features (ensures one-hot encoding consistency)
- Return FeatureSet (not dict) from `step()`

### Handling Missing Values

If a feature might be absent:

```python
from reil.datatypes.feature import NoneFeature

# In step():
if patient_genotype_known:
    state['genotype'] = Feature.categorical('genotype', value=genotype, categories=(…))
else:
    state['genotype'] = NoneFeature('genotype')  # Graceful degradation
```

## Configuration Patterns

### Hierarchical Config with YAML Anchors

```yaml
# Define a reusable learner config
q_learner: &q_learner
  reil.learners.q_learner.QLearner:
    model:
      reil.learners.q_learner.DeepQModel:
        learning_rate: 1e-3
        hidden_layer_sizes: [256, 128, 64]
    buffer:
      reil.datatypes.buffers.vanilla_experience_replay.VanillaExperienceReplay:
        buffer_size: 500
        batch_size: 64

# Use it in multiple agent configs
agent_1:
  reil.agents.DeepQLearning:
    name: agent_1
    learner:
      <<: *q_learner  # Inherits learner config

agent_2:
  reil.agents.DeepQLearning:
    name: agent_2
    learner:
      <<: *q_learner  # Reuses same config
      # Can override specific values if needed
```

### Variable Substitution

ReiL config loading supports `$variable$` placeholders:

```yaml
# Define variables
my_agent:
  reil.agents.DeepQLearning:
    name: $experiment_name$_agent
    learner:
      reil.learners.q_learner.QLearner:
        tensorboard_filename: $h$_$action$$sdemon$
        tensorboard_path: $log_path$
```

Then pass substitution dict during instantiation:
```python
from reil.utils.yaml_tools import load_config_dict

config = load_config_dict(
    'agents.yaml',
    variable_dict={
        'experiment_name': 'exp_001',
        'log_path': './logs',
        'h': 'hidden_layer'
    }
)
```

### Lambda Expressions for Dynamic Parameters

```yaml
exploration_strategy:
  reil.utils.exploration_strategies.VariableEpsilonGreedy:
    epsilon: "lambda n: 1/(1+n)"  # Decaying epsilon over iterations
```

This creates an epsilon-greedy explorer that reduces exploration as training progresses.

## Experiment Orchestration Workflow

### Typical Experiment Flow

1. **Define Subject(s)**: Patient populations, environment models, etc.
2. **Define Agent(s)**: Learning algorithms, exploration strategies
3. **Define Environment**: How agents interact with subjects (Sequential, Single, Task)
4. **Configure Trajectory Dumper**: Record (state, action, reward) for analysis
5. **Run Experiment**: Execute environment, step agents/subjects, collect results
6. **Analyze Results**: Compute statistics, visualize learning curves

### Example: Warfarin Dosing Experiment

See [warfarin_dosing/wd_utils/experiment.py](warfarin_dosing/wd_utils/experiment.py) for full example.

```python
from reil import subjects, agents, environments, learners
from reil.healthcare.subjects import Warfarin
from reil.healthcare.agents import WarfarinAgent

# 1. Create subject (patient population)
patient = PatientWarfarinRavvaz(model=HambergPKPD(randomized=True))
subject = Warfarin(patient=patient)

# 2. Create learnable agent
agent = DeepQLearning(
    learner=QLearner(...),
    exploration_strategy=VariableEpsilonGreedy(epsilon=0.1),
    discount_factor=0.95
)

# 3. Create environment to orchestrate interaction
env = Sequential(subjects=[subject], agents=[agent])

# 4. Run training loop
for episode in range(num_episodes):
    env.step()  # Executes one subject step
    
# 5. Analyze
agent.save('my_trained_agent')  # Via ReilBase.save()
```

## Common Workflow Patterns

### Pattern 1: Comparing Multiple Agents on Same Subject

```yaml
# agents.yaml
baseline_dqn: &dqn_config
  reil.agents.DeepQLearning:
    learner: ...

baseline_ppo: &ppo_config
  reil.agents.PPO:
    learner: ...

baseline_protocol:
  reil.healthcare.agents.WarfarinAgent:
    study_arm: AAA
```

Run experiments with each agent, compare results in [warfarin_dosing/experiments/analysis.ipynb](warfarin_dosing/experiments/analysis.ipynb).

### Pattern 2: Hyperparameter Sweeps

Define multiple learner configs with different hyperparameters:

```yaml
dqn_lr_1e3: &dqn_lr_1e3
  learning_rate: 1e-3

dqn_lr_1e4: &dqn_lr_1e4
  learning_rate: 1e-4

agent_lr_1e3:
  reil.agents.DeepQLearning:
    learner:
      <<: *dqn_lr_1e3

agent_lr_1e4:
  reil.agents.DeepQLearning:
    learner:
      <<: *dqn_lr_1e4
```

### Pattern 3: Using Demons for Research Interventions

Demons modify agent or subject behavior for ablation studies:

```python
# Subject demon: remove genetic features for first N days
subject_demon = SubjectDemon(
    state_modifier=Modifier(
        name='no_genotypes',
        condition=lambda x: x['day'].value < 7
    )
)

# Agent demon: use random actions for exploration
agent_demon = AgentDemon(
    sub_agent=RandomAgent(),
    condition=lambda x, _: episode < 10
)
```

## Troubleshooting Common Issues

### Issue: Agent Doesn't Learn / Reward is Always 0

**Diagnosis**:
- Check reward computation in `Subject.step()` — is it returning non-zero values?
- Verify state features are being properly populated
- Check if `training_trigger` is set to `'none'` (should not be)

**Fix**:
```python
def step(self, action):
    # Ensure reward is computed
    reward = self._compute_clinical_outcome()  # Must be non-zero sometimes
    new_state = self._update_patient_state(action)
    return new_state, reward
```

### Issue: Neural Network Diverges During Training

**Diagnosis**:
- Features might not be normalized (values outside typical NN input range)
- Learning rate too high
- Batch size too small

**Fix**:
```python
# Ensure all numerical features are bounded
Feature.numerical('value', value=50, lower=0, upper=100, normalized=(0.5,))
# Reduce learning_rate in config, increase batch_size
```

### Issue: Missing Value MISSING Causing Crashes

**Diagnosis**:
- Feature marked as missing but downstream code expects valid value

**Fix**:
```python
# Explicitly handle missing values in feature extraction
if state['genotype'].value == MISSING:
    return some_default_embedding()
```

### Issue: Old Serialized Agents Won't Load

**Diagnosis**:
- Class was renamed or moved, CustomUnPickler doesn't have mapping

**Fix**:
- Update [reil/serialization.py](reil/serialization.py) with the class rename mapping
- Re-save the agent with new code

## Recommended Workflow for Extensions

### Adding a Custom Subject for Your Domain

1. **Create subject class** inheriting from `Subject`:
   - Define state in `__init__` via `self.state.add_definition()`
   - Implement `step(action) -> (state_featureset, reward)`
   - Add domain-specific logic (PK/PD models, game rules, etc.)

2. **Test locally**:
   ```python
   subj = MySubject()
   action = FeatureSet({'dosing': Feature.categorical('dose', value='high', categories=(...))})
   new_state, reward = subj.step(action)
   # Verify new_state is FeatureSet with all defined features
   ```

3. **Define YAML config** for your subject:
   ```yaml
   my_subject:
     my_package.MySubject:
       param1: value1
       param2: value2
   ```

4. **Integrate with agents** in experiment setup (see [warfarin_dosing/wd_utils/experiment.py](warfarin_dosing/wd_utils/experiment.py)).

### Adding a Custom Learner

1. **Create learner class** inheriting from `Learner`:
   - Implement `learn(inputs, targets) -> None`
   - Manage model checkpoints via `save_checkpoint()` / `load_checkpoint()`

2. **Test with simple agent**:
   ```python
   learner = MyLearner(...)
   agent = Agent(learner=learner, exploration_strategy=...)
   # Run a few training steps, verify learn() is called
   ```

3. **Define YAML config** for your learner and integrate into agent configs.

## Quick Reference: File Locations

| Need | Location |
|------|----------|
| Core ReiL architecture | [reil/](reil/) |
| Agent implementations | [reil/agents/](reil/agents/) |
| Subject implementations | [reil/subjects/](reil/subjects/) |
| Learner/QLearner | [reil/learners/](reil/learners/) |
| Feature/FeatureSet | [reil/datatypes/feature.py](reil/datatypes/feature.py) |
| YAML config loading | [reil/utils/yaml_tools.py](reil/utils/yaml_tools.py) |
| Exploration strategies | [reil/utils/exploration_strategies.py](reil/utils/exploration_strategies.py) |
| Warfarin domain code | [reil/healthcare/](reil/healthcare/) |
| Warfarin experiments | [warfarin_dosing/](warfarin_dosing/) |
| Example configs | [warfarin_dosing/configs/](warfarin_dosing/configs/) |
| Experiment orchestration | [warfarin_dosing/wd_utils/experiment.py](warfarin_dosing/wd_utils/experiment.py) |

## See Also

- **[.instructions.md](.instructions.md)** — Deep dive into ReiL architecture and implementation patterns
- **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)** — Design rationale and extension points
- **[.skills/YAML_CONFIG_MASTERY.md](.skills/YAML_CONFIG_MASTERY.md)** — Advanced config patterns and debugging
- **[.skills/EXPERIMENT_ORCHESTRATION.md](.skills/EXPERIMENT_ORCHESTRATION.md)** — Setting up multi-agent/subject experiments
- **[warfarin_dosing/.instructions.md](warfarin_dosing/.instructions.md)** — Warfarin dosing project specifics (entry points, workflow)
