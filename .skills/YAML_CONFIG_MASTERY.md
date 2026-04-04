# YAML Config Mastery: Working with ReiL Configurations

This skill guide teaches you how to author, debug, and optimize YAML configurations for ReiL experiments. It's essential reading if you're setting up experiments or modifying agent/subject parameters.

## Prerequisites

- Familiarity with YAML syntax (key-value, nesting, anchors)
- Basic understanding of ReiL abstractions (Agent, Subject, Learner) from [copilot-instructions.md](../copilot-instructions.md)
- Ability to edit YAML files (any text editor)

## What Is YAML Configuration in ReiL?

YAML configs allow you to:
1. Define agents, subjects, learners, buffers, and interaction protocols
2. Specify hyperparameters (learning rates, layer sizes, buffer sizes, etc.)
3. Compose complex objects hierarchically without touching Python code
4. Reuse configurations across experiments via anchors

**Example: DQN Agent Config** (from [warfarin_dosing/configs/agents.yaml](../../warfarin_dosing/configs/agents.yaml)):

```yaml
dqn_agent_setting: &dqn_agent_setting
  reil.agents.deep_q_learning.DeepQLearning:
    name: deep_q
    learner:
      reil.learners.q_learner.QLearner:
        model:
          reil.learners.q_learner.DeepQModel:
            learning_rate: 1e-3
            hidden_layer_sizes: [256, 128, 64, 32]
        tensorboard_filename: $h$_$action$$sdemon$
        tensorboard_path: ./logs
    buffer:
      reil.datatypes.buffers.vanilla_experience_replay.VanillaExperienceReplay:
        buffer_size: 450
        batch_size: 50
    exploration_strategy:
      reil.utils.exploration_strategies.VariableEpsilonGreedy:
        epsilon: "lambda n: 1/(1+n)"
    discount_factor: 0.95
```

## YAML Structure Basics

### Object Instantiation: Fully-Qualified Class Names

Every YAML node that creates a ReiL object uses a fully-qualified class name as the key:

```yaml
# Structure: YAML_ANCHOR: &optional_anchor_name
#   fully.qualified.ClassName:
#     param1: value1
#     param2: value2

my_agent: &my_agent
  reil.agents.DeepQLearning:
    name: agent_1
    discount_factor: 0.95
    learner:
      reil.learners.q_learner.QLearner:
        learning_rate: 1e-3
```

**How it works**: When ReiL loads this YAML:
1. Finds the fully-qualified class name (e.g., `reil.agents.DeepQLearning`)
2. Imports the class dynamically
3. Recursively instantiates nested objects (learner, model, buffer, etc.)
4. Calls `DeepQLearning(name='agent_1', discount_factor=0.95, learner=<learner_obj>)`

**Key Point**: The class name must match the actual Python import path. Common mistakes:
- `ReiL.agents.DeepQLearning` (wrong: capital R in ReiL)
- `reil.agents.deep_q.DeepQLearning` (wrong: module name, try `deep_q_learning`)

Find the correct path by:
```python
# In Python
from reil.agents import DeepQLearning
print(DeepQLearning.__module__)  # Output: reil.agents.deep_q_learning
# Full path: reil.agents.deep_q_learning.DeepQLearning
```

### YAML Anchors for Config Reuse

Use `&anchor_name` to define, then `<<: *anchor_name` to reuse:

```yaml
# Define once
q_learner_base: &q_learner_base
  reil.learners.q_learner.QLearner:
    learning_rate: 1e-3
    model:
      reil.learners.q_learner.DeepQModel:
        hidden_layer_sizes: [256, 128, 64]

# Reuse in multiple agents
agent_1:
  reil.agents.DeepQLearning:
    name: agent_1
    learner:
      <<: *q_learner_base

agent_2:
  reil.agents.DeepQLearning:
    name: agent_2
    learner:
      <<: *q_learner_base

# Reuse and override
agent_3:
  reil.agents.DeepQLearning:
    name: agent_3
    learner:
      <<: *q_learner_base
      learning_rate: 2e-3  # Override base value
```

**Benefit**: Keeps configs DRY (Don't Repeat Yourself); experiment with new hyperparameters in one place.

### Nested Object Hierarchies

ReiL objects are often nested: Agent → Learner → Model, Agent → Buffer, Agent → ExplorationStrategy.

```yaml
agent:
  reil.agents.DeepQLearning:
    # Top-level params
    name: my_dqn
    discount_factor: 0.95
    
    # Nested learner (+ nested model inside learner)
    learner:
      reil.learners.q_learner.QLearner:
        learning_rate: 1e-3  # Learner param
        model:
          reil.learners.q_learner.DeepQModel:
            hidden_layer_sizes: [256, 128]  # Model param
            output_size: 7
    
    # Nested buffer
    buffer:
      reil.datatypes.buffers.vanilla_experience_replay.VanillaExperienceReplay:
        buffer_size: 500
        batch_size: 64
    
    # Nested exploration strategy
    exploration_strategy:
      reil.utils.exploration_strategies.VariableEpsilonGreedy:
        epsilon: "lambda n: 0.5 / (1 + n)"
```

When ReiL loads this:
1. Creates DeepQModel with hidden_layer_sizes=[256, 128]
2. Creates QLearner with model=<DeepQModel>, learning_rate=1e-3
3. Creates VanillaExperienceReplay with buffer_size=500
4. Creates VariableEpsilonGreedy with epsilon=<lambda>
5. Creates DeepQLearning with learner=<QLearner>, buffer=<buffer_obj>, etc.

## Common Config Patterns

### Pattern 1: Deep Q-Learning Agent (Discrete Actions)

Use when: Discrete action space (e.g., 7 dosing options), value-based learning.

```yaml
dqn_agent: &dqn_config
  reil.agents.deep_q_learning.DeepQLearning:
    name: dqn_agent
    discount_factor: 0.95
    
    learner:
      reil.learners.q_learner.QLearner:
        model:
          reil.learners.q_learner.DeepQModel:
            learning_rate: 1e-3
            hidden_layer_sizes: [256, 128, 64, 32]
            output_size: 7  # Number of discrete actions
        tensorboard_path: ./logs
        tensorboard_filename: dqn_training
    
    buffer:
      reil.datatypes.buffers.vanilla_experience_replay.VanillaExperienceReplay:
        buffer_size: 450
        batch_size: 50
    
    exploration_strategy:
      reil.utils.exploration_strategies.VariableEpsilonGreedy:
        epsilon: "lambda n: 1/(1+n)"  # Decaying exploration
```

**Key parameters to tune**:
- `learning_rate`: Start with 1e-3, reduce if diverging
- `hidden_layer_sizes`: Deeper (more layers) for complex patterns, shallower for simple ones
- `buffer_size`: Larger = more stable but more memory; start with 500
- `batch_size`: Larger = less noisy gradients but slower updates; start with 64
- `epsilon`: Decay schedule; `lambda n: 1/(1+n)` decays as training advances

### Pattern 2: PPO Agent (Policy Gradient)

Use when: Continuous or discrete actions, want stable training, or value estimates are unreliable.

```yaml
ppo_agent: &ppo_config
  reil.agents.ppo.PPO:
    name: ppo_agent
    discount_factor: 0.95
    
    learner:
      reil.learners.ppo_learner.PPOLearner:
        model:
          reil.learners.ppo_learner.DeepPPOModel:
            learning_rate:
              tensorflow.keras.optimizers.schedules.ExponentialDecay:
                initial_learning_rate: 0.01
                decay_steps: 100
                decay_rate: 0.96
                staircase: True
            shared_layer_sizes: [256, 128, 64, 32]
            actor_layer_sizes: [64, 32]
            critic_layer_sizes: [16]
            entropy_loss_coef: 0.01
        tensorboard_path: ./logs
        tensorboard_filename: ppo_training
    
    exploration_strategy:
      reil.utils.exploration_strategies.ConstantEpsilonGreedy:
        epsilon: 0.0  # PPO doesn't use epsilon; set to 0
```

**Key parameters**:
- `learning_rate`: Can be constant or schedule (ExponentialDecay shown above)
- `shared_layer_sizes`: Layers shared between actor and critic
- `actor_layer_sizes`: Actor-specific layers
- `critic_layer_sizes`: Critic-specific layers
- `entropy_loss_coef`: Encourages exploration (higher = more exploration)

### Pattern 3: A2C Agent (Actor-Critic)

Use when: Want lower-variance policy gradients than vanilla policy gradient.

```yaml
a2c_agent: &a2c_config
  reil.agents.A2C:
    name: a2c_agent
    discount_factor: 0.95
    
    learner:
      reil.learners.A2CLearner:
        model:
          reil.learners.actor_critic_learner.DeepA2CModel:
            output_lengths: [31, 7]  # [num_samples, num_actions]
            learning_rate:
              tensorflow.keras.optimizers.schedules.ExponentialDecay:
                initial_learning_rate: 0.01
                decay_steps: 100
                decay_rate: 0.96
                staircase: True
            shared_layer_sizes: [256, 128, 64, 32]
        tensorboard_path: ./logs
        tensorboard_filename: a2c_training
```

### Pattern 4: Baseline / Reference Agent

Use for comparing against established baselines.

```yaml
baseline_aaa:
  reil.healthcare.agents.WarfarinAgent:
    study_arm: AAA

baseline_caa:
  reil.healthcare.agents.WarfarinAgent:
    study_arm: CAA

baseline_pgaa:
  reil.healthcare.agents.WarfarinAgent:
    study_arm: PGAA
```

**Available study_arm values** (warfarin protocols):
- `AAA`: Age-adjusted algorithm
- `CAA`: Clinical age-adjusted
- `PGAA`: Pharmacogenetic-adjusted
- `PGPGI`: Pharmacogenetic + patient INR
- `PGPGA`: Pharmacogenetic + patient + genetic

## Variable Substitution

ReiL supports `$variable_name$` placeholders in YAML, which are replaced at load time.

### Example

```yaml
# agents.yaml
my_agent:
  reil.agents.DeepQLearning:
    name: $experiment_id$_agent
    learner:
      reil.learners.q_learner.QLearner:
        tensorboard_filename: $experiment_id$_$timestamp$
        tensorboard_path: $log_dir$
```

**Load with substitution**:

```python
from reil.utils.yaml_tools import load_config_dict
import datetime

config = load_config_dict(
    'agents.yaml',
    variable_dict={
        'experiment_id': 'exp_001',
        'timestamp': datetime.datetime.now().strftime('%Y%m%d_%H%M%S'),
        'log_dir': './logs/exp_001'
    }
)

agent = config['my_agent']  # name is now 'exp_001_agent'
```

**Common variables**:
- `$h$`: Hidden layer identifier
- `$action$`: Action index
- `$sdemon$`: Subject demon identifier
- `$experiment_name$`: Unique experiment ID
- `$project_path$`: Project root directory
- `$log_path$`: Path to save logs

## Lambda Expressions for Functions

Some parameters (especially exploration strategies) accept callables. Use `"lambda ..."` syntax:

```yaml
exploration_strategy:
  reil.utils.exploration_strategies.VariableEpsilonGreedy:
    epsilon: "lambda n: 1/(1+n)"  # Decays over iterations n

# Another example: step-based decay
epsilon: "lambda n: 0.5 if n < 1000 else 0.1"
```

**Important**: Always quote lambda expressions (e.g., `"lambda ..."`), so YAML parses them as strings. ReiL evaluates them at runtime.

## Config Debugging & Validation

### Issue 1: Class Not Found

**Error**: `ImportError: cannot import name 'DeepQLearning' from 'reil.agents'`

**Cause**: Incorrect fully-qualified class name in YAML.

**Fix**:
1. Find the correct path:
   ```python
   from reil.agents import DeepQLearning
   print(DeepQLearning.__module__)  # reil.agents.deep_q_learning
   # Use: reil.agents.deep_q_learning.DeepQLearning
   ```
2. Update YAML

### Issue 2: Missing Required Parameter

**Error**: `TypeError: __init__() missing required keyword argument 'learning_rate'`

**Cause**: Forgot to specify a required parameter in YAML.

**Fix**:
1. Check the class `__init__()` signature:
   ```python
   from reil.learners.q_learner import QLearner
   import inspect
   print(inspect.signature(QLearner.__init__))
   ```
2. Add missing param to YAML

### Issue 3: Invalid Parameter Value

**Error**: `ValueError: learning_rate must be positive`

**Cause**: Parameter value invalid (e.g., negative learning rate).

**Fix**: Review parameter bounds in docs and YAML. Common bounds:
- `learning_rate`: Positive float (e.g., 1e-5 to 1e-2)
- `buffer_size`: Positive int (e.g., 100 to 10000)
- `batch_size`: Positive int, typically ≤ buffer_size
- `hidden_layer_sizes`: List of positive ints (e.g., [256, 128, 64])

### Debugging: Print Loaded Config

After loading, print to verify:

```python
from reil.utils.yaml_tools import load_config_dict
import json

config = load_config_dict('agents.yaml')
agent = config['my_agent']

# Print agent attributes
print(f"Agent name: {agent._name}")
print(f"Discount factor: {agent._discount_factor}")
print(f"Learner: {agent._learner}")
```

### Debugging: Enable Logging

To see what's happening during config load:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

config = load_config_dict('agents.yaml')
# Now you'll see debug messages from yaml_tools
```

## Best Practices

### 1. Use Anchors for Common Configurations

❌ **Bad**: Duplicate learner config in every agent

```yaml
agent_1:
  reil.agents.DeepQLearning:
    learner:
      reil.learners.q_learner.QLearner:
        learning_rate: 1e-3
        
agent_2:
  reil.agents.DeepQLearning:
    learner:
      reil.learners.q_learner.QLearner:
        learning_rate: 1e-3  # Duplicated!
```

✅ **Good**: Define once, reuse

```yaml
base_learner: &base_learner
  reil.learners.q_learner.QLearner:
    learning_rate: 1e-3

agent_1:
  reil.agents.DeepQLearning:
    learner:
      <<: *base_learner

agent_2:
  reil.agents.DeepQLearning:
    learner:
      <<: *base_learner
```

### 2. Document Hyperparameter Tuning

Add comments explaining key parameters:

```yaml
my_agent:
  reil.agents.DeepQLearning:
    # Agent-level params
    discount_factor: 0.95  # Weight future rewards; higher = more forward-looking
    
    # Learner configuration
    learner:
      reil.learners.q_learner.QLearner:
        learning_rate: 1e-3  # Neural network learning rate; reduce if diverging
        model:
          reil.learners.q_learner.DeepQModel:
            hidden_layer_sizes: [256, 128, 64]  # Increase for complex patterns
```

### 3. Organize Configs by Purpose

Create separate YAML files for clarity:

```
configs/
  agents.yaml         # Agent/learner configs
  subjects.yaml       # Subject/patient configs
  tasks.yaml          # Task definitions
  sessions.yaml       # Session setup (multi-agent/subject)
  interaction_protocols.yaml  # How agents interact
```

### 4. Version Control Your Configs

Track configs in git so you can reproduce experiments:

```bash
git add configs/
git commit -m "Add DQN baseline config for exp_001"
```

Then reference the git hash when reporting results:

```
Results: exp_001
Config commit: a1b2c3d
Config file: configs/agents.yaml::dqn_agent
```

## Advanced: Custom Parameter Types

Some parameters support complex types not easily expressed in YAML.

### TensorFlow Learning Rate Schedules

```yaml
learning_rate:
  tensorflow.keras.optimizers.schedules.ExponentialDecay:
    initial_learning_rate: 0.01
    decay_steps: 100
    decay_rate: 0.96
    staircase: True
```

This creates a TensorFlow ExponentialDecay schedule that decays learning rate.

### Custom Python Objects

If you need a custom Python object (non-ReiL class), instantiate in Python code:

```python
# In Python, not YAML
from my_module import CustomObject

config = load_config_dict('agents.yaml')
agent = config['my_agent']
agent.custom_param = CustomObject(...)
```

## Complete Example: Multi-Algorithm Comparison

```yaml
# Compare DQN, PPO, A2C on same subject

base_exploration: &base_exploration
  reil.utils.exploration_strategies.VariableEpsilonGreedy:
    epsilon: "lambda n: 1/(1+n)"

# DQN
dqn_agent: &dqn_agent
  reil.agents.deep_q_learning.DeepQLearning:
    name: dqn_agent
    discount_factor: 0.95
    learner:
      reil.learners.q_learner.QLearner:
        learning_rate: 1e-3
        model:
          reil.learners.q_learner.DeepQModel:
            hidden_layer_sizes: [256, 128, 64]
            output_size: 7
    buffer:
      reil.datatypes.buffers.vanilla_experience_replay.VanillaExperienceReplay:
        buffer_size: 450
        batch_size: 50
    exploration_strategy:
      <<: *base_exploration

# PPO
ppo_agent: &ppo_agent
  reil.agents.ppo.PPO:
    name: ppo_agent
    discount_factor: 0.95
    learner:
      reil.learners.ppo_learner.PPOLearner:
        learning_rate: 0.01
        model:
          reil.learners.ppo_learner.DeepPPOModel:
            shared_layer_sizes: [256, 128, 64]
    exploration_strategy:
      reil.utils.exploration_strategies.ConstantEpsilonGreedy:
        epsilon: 0.0

# A2C
a2c_agent: &a2c_agent
  reil.agents.A2C:
    name: a2c_agent
    discount_factor: 0.95
    learner:
      reil.learners.A2CLearner:
        learning_rate: 0.01
        model:
          reil.learners.actor_critic_learner.DeepA2CModel:
            shared_layer_sizes: [256, 128, 64]

# Baseline protocols
baseline_aaa:
  reil.healthcare.agents.WarfarinAgent:
    study_arm: AAA
```

Then in Python:

```python
from reil.utils.yaml_tools import load_config_dict

config = load_config_dict('agents.yaml')

agents = {
    'DQN': config['dqn_agent'],
    'PPO': config['ppo_agent'],
    'A2C': config['a2c_agent'],
    'AAA_Baseline': config['baseline_aaa']
}

# Run experiments with each agent
for agent_name, agent in agents.items():
    print(f"Training {agent_name}...")
    # Training code here
```

## Reference: Config File Locations

In warfarin_dosing project:
- [warfarin_dosing/configs/agents.yaml](../../warfarin_dosing/configs/agents.yaml) — Agent definitions (DQN, PPO, A2C, baselines)
- [warfarin_dosing/configs/subjects.yaml](../../warfarin_dosing/configs/subjects.yaml) — Patient/subject configs
- [warfarin_dosing/configs/tasks.yaml](../../warfarin_dosing/configs/tasks.yaml) — Task definitions
- [warfarin_dosing/configs/sessions.yaml](../../warfarin_dosing/configs/sessions.yaml) — Session setups

## See Also

- [copilot-instructions.md](../copilot-instructions.md) — High-level guidance and workflow patterns
- [.skills/EXPERIMENT_ORCHESTRATION.md](EXPERIMENT_ORCHESTRATION.md) — Running multi-entity experiments
- [ARCHITECTURE.md](../docs/ARCHITECTURE.md) — Design principles and extension points
- Code: [reil/utils/yaml_tools.py](../reil/utils/yaml_tools.py) — Config loading implementation
