# Experiment Orchestration: Setting Up & Running Multi-Agent/Subject Experiments

This skill guide teaches you how to orchestrate RL experiments with multiple agents and subjects. It's essential for running large-scale research studies and understanding how ReiL coordinates interaction between entities.

## Prerequisites

- Understanding of ReiL abstractions (Agent, Subject, Environment) from [copilot-instructions.md](../copilot-instructions.md)
- Familiarity with YAML configuration from [.skills/YAML_CONFIG_MASTERY.md](YAML_CONFIG_MASTERY.md)
- Basic Python knowledge

## What Is Experiment Orchestration?

Experiment orchestration means:
1. **Creating entities** (agents, subjects) from configs or code
2. **Registering relationships** (which agents interact with which subjects)
3. **Setting up monitoring** (trajectories, statistics, checkpoints)
4. **Running training loops** (orchestrating step-by-step interaction)
5. **Collecting & aggregating results**

**Real-world example**: Research project comparing 3 RL agents vs. 5 baseline protocols on 1000 patient simulations, collecting statistics and trajectories for later analysis.

## Typical Experiment Flow

```
┌─ Load Configurations (YAML)
│    agents.yaml, subjects.yaml, tasks.yaml, sessions.yaml
├─ Create Subjects (Patient populations, environments)
├─ Create Agents (DQN, PPO, A2C, baselines)
├─ Register Relationships (Subjects know their agents, vice versa)
├─ Set Up Monitoring (Trajectory dumpers, loggers, checkpoints)
├─ Configure Environment (Sequential, Single, Task)
├─ Run Training Loop (Step subjects, triggers agent learning)
├─ Collect Results (Statistics, trajectories, models)
└─ Analyze & Report
```

## Step 1: Load Configurations

Most experiments start with YAML configs rather than hardcoding in Python.

```python
from reil.utils.yaml_tools import load_config_dict

# Load all configs
agents_config = load_config_dict('configs/agents.yaml')
subjects_config = load_config_dict('configs/subjects.yaml')
tasks_config = load_config_dict('configs/tasks.yaml')

# Extract specific entities
dqn_agent_template = agents_config['dqn_agent']
ppo_agent_template = agents_config['ppo_agent']
warfarin_subject_template = subjects_config['warfarin_subject']
```

**Best Practice**: Load configs with variable substitution for experiment IDs:

```python
import datetime

experiment_id = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

agents_config = load_config_dict(
    'configs/agents.yaml',
    variable_dict={
        'experiment_name': experiment_id,
        'log_path': f'./logs/{experiment_id}',
        'project_path': './'
    }
)
```

## Step 2: Create Subjects

Subjects represent the environment or system the agents interact with.

### Single Subject, Multiple Agents

```python
from reil import subjects
from reil.healthcare.subjects import Warfarin
from reil.healthcare import PatientWarfarinRavvaz, HambergPKPD

# Create one subject (patient simulation)
patient = PatientWarfarinRavvaz(
    model=HambergPKPD(randomized=True, cache_size=90)
)
subject = Warfarin(patient=patient, name='patient_1')

# Multiple agents will interact with this patient
```

### Multiple Subjects (Subject Population)

```python
# Create multiple subject instances for population-based experiments
num_patients = 100
subjects_list = []

for i in range(num_patients):
    patient = PatientWarfarinRavvaz(
        model=HambergPKPD(randomized=True, cache_size=90)
    )
    subject = Warfarin(patient=patient, name=f'patient_{i:04d}')
    subjects_list.append(subject)

print(f"Created {len(subjects_list)} patient simulations")
```

**Best Practice**: Use subject generators for scalability:

```python
from reil.utils import SubjectGenerator, InstanceGeneratorBatch

# Create a batch generator (lazy instantiation)
subject_generator = SubjectGenerator(
    cls=Warfarin,
    args_generator=lambda i: {
        'patient': PatientWarfarinRavvaz(model=HambergPKPD(randomized=True)),
        'name': f'patient_{i:04d}'
    },
    start=0,
    stops=100
)

# Subjects are created on-demand, not all at once
```

## Step 3: Create Agents

Agents are instantiated from configs or code.

### Creating from Config

```python
# Load predefined agent configs
agents_config = load_config_dict('configs/agents.yaml')

dqn_agent = agents_config['dqn_agent']
ppo_agent = agents_config['ppo_agent']
baseline_aaa = agents_config['baseline_aaa']

# Agents are ready to train/act
```

### Creating Multiple Agent Instances

For population-level experiments, often you want one trained agent applied to multiple patients:

```python
# Create one trained DQN agent
dqn_agent = agents_config['dqn_agent']

# Apply to multiple patients
for subject in subjects_list:
    # During training, each patient gets interacted with by the same agent
    # (agent learns from all patients collectively)
    pass
```

Or, train separate agents per subject:

```python
# Each patient gets its own agent (rarely done, usually for comparative analysis)
agents_per_subject = {}
for i, subject in enumerate(subjects_list):
    agent = agents_config['dqn_agent']
    agent._name = f'dqn_patient_{i:04d}'
    agents_per_subject[subject._name] = agent
```

### Creating Custom Agents in Code

If you don't have a config, instantiate directly:

```python
from reil.agents import DeepQLearning
from reil.learners.q_learner import QLearner, DeepQModel
from reil.datatypes.buffers import VanillaExperienceReplay
from reil.utils.exploration_strategies import VariableEpsilonGreedy

# Build learner
model = DeepQModel(learning_rate=1e-3, hidden_layer_sizes=[256, 128, 64])
learner = QLearner(model=model)

# Build buffer
buffer = VanillaExperienceReplay(buffer_size=450, batch_size=50)

# Build exploration strategy
exploration = VariableEpsilonGreedy(epsilon=lambda n: 1/(1+n))

# Assemble agent
agent = DeepQLearning(
    name='custom_dqn',
    learner=learner,
    buffer=buffer,
    exploration_strategy=exploration,
    discount_factor=0.95
)
```

## Step 4: Register Relationships

Agents and subjects need to know about each other for:
- Statistics tracking per entity (e.g., agent 1's win rate vs. agent 2's)
- Partial observability (agent sees subject's state; subject knows agent's identity)
- Demons (targeted modifications to specific entities)

```python
# Subject registers agents (subjects care who's observing/acting)
for agent in [dqn_agent, ppo_agent, baseline_aaa]:
    subject.register(agent)

# Now subject can provide agent-specific statistics
dqn_stats = subject.statistic(dqn_agent._id, 'episode_reward')
```

**In multi-agent scenarios**:

```python
# One subject, three agents
subject = subjects_list[0]

agents = [dqn_agent, ppo_agent, baseline_aaa]
for agent in agents:
    subject.register(agent)
    agent.register(subject)  # Agents also know about the subject

# Now statistics are tracked per agent
for agent in agents:
    stats = subject.statistic(agent._id, 'avg_reward')
    print(f"{agent._name}: {stats}")
```

## Step 5: Set Up Monitoring (Trajectories & Logging)

Trajectories are (state, action, reward) sequences recorded during interaction. Essential for later analysis.

### Using Trajectory Dumpers

Subjects can optionally dump trajectories to disk:

```python
from reil.healthcare.trajectory_dumper import TrajectoryDumper

# Create trajectory dumper
dumper = TrajectoryDumper(
    filename='trajectory',
    path=f'./results/{experiment_id}/trajectories'
)

# Attach to subject
subject = Warfarin(
    patient=patient,
    state_dumper=dumper,
    name='patient_1'
)

# Now interactions are automatically logged to disk
# Later analysis can read: ./results/{experiment_id}/trajectories/trajectory*
```

### Logging & Checkpointing

ReiL manages logging automatically:

```python
# Agent logger
agent._logger.info(f"Training episode {episode}")
agent._logger.debug(f"State: {state}")

# Save agent checkpoint periodically
if (episode + 1) % 100 == 0:
    agent.save(f'checkpoints/agent_ep_{episode:06d}')
    print(f"Saved checkpoint at episode {episode}")

# Load from checkpoint
agent.load('checkpoints/agent_ep_000500')
```

## Step 6: Configure Environment

The Environment orchestrates step-by-step interaction between agents and subjects.

### Sequential Environment (Multiple Subjects)

Use when: Multiple subjects, one agent per step.

```python
from reil.environments import Sequential

# Create environment for population-level training
env = Sequential(
    subjects=subjects_list,  # 100 patients
    agents=[dqn_agent],       # 1 DQN agent (learns from all)
    name='population_training'
)

# Step method: selects next subject, gets its state, agent acts, subject steps
for episode in range(num_episodes):
    env.step()  # One call = one interaction (state, action, reward, next_state)
    
    if (episode + 1) % 1000 == 0:
        print(f"Episode {episode + 1}")
        dqn_agent.save(f'checkpoints/episode_{episode:06d}')
```

### Single Environment (One Subject, Multiple Episodes)

Use when: Single subject, run many episodes, collect statistics.

```python
from reil.environments import Single

subject = subjects_list[0]
agent = dqn_agent

env = Single(
    subject=subject,
    agent=agent,
    name='single_patient_training'
)

for episode in range(100):
    env.step()

# Get statistics from subject
stats = subject.statistic(agent._id, 'episode_reward')
print(f"Agent average reward: {stats}")
```

### Task Environment (Subject-Specific Completion)

Use when: Training toward specific goals (e.g., maintain INR in therapeutic range for N days).

```python
from reil.environments import Task

# Define task: maintain INR in range for 30 days, then episode ends
def task_fn(subject, episode_num):
    return subject._patient.simulation_day > 30

env = Task(
    subject=subject,
    agent=agent,
    termination_fn=task_fn,
    name='task_maintenance'
)

for episode in range(100):
    env.step()
```

## Step 7: Run Training Loop with Learning Triggers

Agents learn when triggered by `training_trigger` (specified in agent config).

### Training Trigger Options

```python
# Trigger='termination': Learn after episode ends
agent = agents_config['dqn_agent']  # Typically has training_trigger='termination'
for episode in range(100):
    env.step()
    # Agent only learns after env.step() completes an episode (subject resets)

# Trigger='reward': Learn after every reward signal
agent._training_trigger = 'reward'
for episode in range(100):
    env.step()
    # Agent learns multiple times per episode (after every reward)

# Trigger='state': Learn after every state observation
agent._training_trigger = 'state'
for episode in range(100):
    env.step()
    # Agent learns constantly (potentially unstable)

# Trigger='none': Never learn (useful for behavior cloning, rollout collection)
agent._training_trigger = 'none'
for episode in range(100):
    env.step()
    # Agent acts but doesn't learn; can collect data for offline training
```

## Step 8: Multi-Agent Experiment Pattern

Comparing multiple agents on same subject:

```python
from reil.environments import Sequential

# Create one subject (or subject list)
subject = subjects_list[0]

# Create multiple agents
agents_to_compare = {
    'DQN': agents_config['dqn_agent'],
    'PPO': agents_config['ppo_agent'],
    'Baseline_AAA': agents_config['baseline_aaa']
}

results = {}

for agent_name, agent in agents_to_compare.items():
    print(f"\n{'='*50}")
    print(f"Training {agent_name}")
    print(f"{'='*50}")
    
    # Register agent with subject
    subject.register(agent)
    
    # Create environment
    env = Sequential(subjects=[subject], agents=[agent], name=agent_name)
    
    # Train
    for episode in range(1000):
        env.step()
    
    # Collect results
    stats = subject.statistic(agent._id, 'episode_reward')
    results[agent_name] = stats
    
    # Save model
    agent.save(f'results/{agent_name}_trained_model')

# Compare results
for agent_name, stats in results.items():
    print(f"{agent_name}: {stats}")
```

## Step 9: Using Demons for Research Interventions

Demons are modifiers that change agent/subject behavior for ablation studies.

```python
from reil.agents import AgentDemon, SubjectDemon
from reil.subjects import Modifier

# Agent Demon: Use random actions for first N episodes (exploration)
random_exploration_demon = AgentDemon(
    sub_agent=agents_config['random_agent'],
    condition_fn=lambda state, episode: episode < 100
)
agent.add_demon(random_exploration_demon)

# Subject Demon: Remove genetic features for first N days
no_genotypes_demon = SubjectDemon(
    state_modifier=Modifier(
        name='no_genotypes',
        condition=lambda state: state['day'].value < 7
    )
)
subject.add_demon(no_genotypes_demon)

# Now training proceeds with demons active
env = Sequential(subjects=[subject], agents=[agent])
for episode in range(1000):
    env.step()
    # Demons modify behavior automatically based on their conditions
```

## Step 10: Collecting & Aggregating Results

After training, analyze results.

### Reading Statistics

```python
# Get statistics from subject
episode_rewards = subject.statistic(agent._id, 'episode_reward')
avg_reward = subject.statistic(agent._id, 'avg_reward')
win_rate = subject.statistic(agent._id, 'win_rate')

print(f"Agent {agent._name}: avg_reward={avg_reward}, win_rate={win_rate}")
```

### Reading Trajectories

```python
# Trajectories dumped to disk during interaction
import pathlib

trajectory_path = pathlib.Path(f'./results/{experiment_id}/trajectories')
trajectory_files = sorted(trajectory_path.glob('trajectory*'))

# Each file contains (state, action, reward) sequences for one episode
for traj_file in trajectory_files[:5]:  # Read first 5 episodes
    with open(traj_file, 'rb') as f:
        trajectory_data = f.read()  # Typically a pickle or HDF5 file
    print(f"Trajectory from {traj_file}")
```

See [warfarin_dosing/read_results_funcs.py](../../warfarin_dosing/read_results_funcs.py) for utilities.

### Aggregating Results Across Experiment Runs

```python
import csv
import pathlib

# Create results CSV
results_csv = pathlib.Path(f'./results/{experiment_id}/results.csv')
with open(results_csv, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Agent', 'AvgReward', 'WinRate', 'ModelPath'])
    
    for agent_name, agent in agents_to_compare.items():
        stats = subject.statistic(agent._id, 'episode_reward')
        avg_reward = sum(stats) / len(stats) if stats else 0
        win_rate = sum(1 for r in stats if r > 0) / len(stats) if stats else 0
        
        writer.writerow([
            agent_name,
            f'{avg_reward:.4f}',
            f'{win_rate:.4f}',
            f'results/{agent_name}_trained_model'
        ])

print(f"Results saved to {results_csv}")
```

## Complete Example: Multi-Algorithm Comparison Experiment

```python
"""
Experiment: Compare DQN vs PPO vs A2C vs clinical baselines on warfarin dosing
"""

import datetime
import pathlib
from reil.utils.yaml_tools import load_config_dict
from reil.environments import Sequential
from reil.healthcare.subjects import Warfarin
from reil.healthcare import PatientWarfarinRavvaz, HambergPKPD

# Setup
experiment_id = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
result_dir = pathlib.Path(f'./results/{experiment_id}')
result_dir.mkdir(parents=True, exist_ok=True)

# Load configs
print("Loading configurations...")
config = load_config_dict(
    'configs/agents.yaml',
    variable_dict={
        'experiment_name': experiment_id,
        'log_path': str(result_dir / 'logs'),
        'project_path': './'
    }
)

# Create subjects (patient population)
print("Creating 50 patient simulations...")
subjects_list = []
for i in range(50):
    patient = PatientWarfarinRavvaz(model=HambergPKPD(randomized=True))
    subject = Warfarin(patient=patient, name=f'patient_{i:04d}')
    subjects_list.append(subject)

# Setup agents to compare
agents_to_compare = {
    'DQN': config['dqn_agent'],
    'PPO': config['ppo_agent'],
    'A2C': config['a2c_agent'],
    'AAA_Baseline': config['baseline_aaa'],
    'PGAA_Baseline': config['baseline_pgaa']
}

results = {}

# Train each agent
for agent_name, agent in agents_to_compare.items():
    print(f"\n{'='*60}")
    print(f"Training {agent_name} on population...")
    print(f"{'='*60}")
    
    # Register agent with all subjects
    for subject in subjects_list:
        subject.register(agent)
    
    # Create environment
    env = Sequential(subjects=subjects_list, agents=[agent], name=agent_name)
    
    # Train
    num_episodes = 10000
    for episode in range(num_episodes):
        env.step()
        
        if (episode + 1) % 1000 == 0:
            print(f"  Episode {episode + 1}/{num_episodes}")
    
    # Collect results
    avg_reward = 0
    for subject in subjects_list:
        stats = subject.statistic(agent._id, 'episode_reward')
        avg_reward += sum(stats) / len(stats) if stats else 0
    avg_reward /= len(subjects_list)
    
    results[agent_name] = avg_reward
    
    # Save model
    agent.save(result_dir / f'{agent_name}_model')
    print(f"✓ Saved {agent_name} model to {result_dir / f'{agent_name}_model'}")

# Report results
print(f"\n{'='*60}")
print(f"FINAL RESULTS (Experiment {experiment_id})")
print(f"{'='*60}")
for agent_name in sorted(results.keys(), key=lambda x: results[x], reverse=True):
    print(f"{agent_name:20s}: {results[agent_name]:8.4f}")

print(f"\nResults saved to: {result_dir}")
```

## Debugging & Troubleshooting

### Issue: Agent Doesn't Learn

**Check**:
1. `training_trigger` is not `'none'`
2. Subject is returning non-zero rewards in `step(action)`
3. Agent has enough experience (buffer_size reached)

**Fix**:
```python
# Verify training trigger
print(f"Training trigger: {agent._training_trigger}")
if agent._training_trigger == 'none':
    agent._training_trigger = 'termination'  # Change it

# Verify rewards
print(f"Subject reward output: {subject.step(action)[1]}")
```

### Issue: Memory Grows Unbounded

**Cause**: Not clearing old checkpoints, trajectory files, or buffers.

**Fix**:
```python
# Clean up old checkpoints periodically
import shutil
import pathlib

checkpoint_dir = pathlib.Path('checkpoints')
old_checkpoints = sorted(checkpoint_dir.glob('*'))[:-10]  # Keep only 10 latest
for old_cp in old_checkpoints:
    shutil.rmtree(old_cp)
```

### Issue: Results Inconsistent Across Runs

**Cause**: Non-deterministic behavior (random seeds not fixed).

**Fix**:
```python
import random
import numpy as np
import tensorflow as tf

# Set seeds for reproducibility
def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

set_seeds(42)
```

## Reference: Real-World Example

See [warfarin_dosing/wd_utils/experiment.py](../../warfarin_dosing/wd_utils/experiment.py) for full orchestration implementation with:
- Config-based agent/subject creation
- Demon application
- Trajectory dumping
- Multi-process job submission
- Results aggregation

## See Also

- [copilot-instructions.md](../copilot-instructions.md) — Abstraction selection and workflows
- [.skills/YAML_CONFIG_MASTERY.md](YAML_CONFIG_MASTERY.md) — Config patterns and debugging
- [ARCHITECTURE.md](../docs/ARCHITECTURE.md) — Design and extension points
- Code: [reil/environments/](../reil/environments/) — Environment implementations
- Code: [warfarin_dosing/run.py](../../warfarin_dosing/run.py) — Cluster job submission example
