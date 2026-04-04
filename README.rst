ReiL - A Reinforcement Learning Package
=======================================

.. image:: https://img.shields.io/pypi/v/Reil   :alt: PyPI

.. image:: https://img.shields.io/pypi/l/ReiL   :alt: PyPI - License

A hierarchical, extensible reinforcement learning framework for research and production use. ReiL provides reusable abstractions (Agents, Subjects, Environments, Learners) for building and experimenting with RL systems, with built-in support for healthcare domain applications (e.g., warfarin dosing optimization).


Features
--------

- **Hierarchical Architecture**: Clean separation via ReilBase → Stateful → domain-specific classes
- **Configuration-Driven**: YAML-based entity instantiation; swap algorithms without recompilation
- **Feature/FeatureSet**: Universal state representation with automatic normalization, type safety, and missing value handling
- **Multiple Learning Algorithms**: Deep Q-Learning, PPO, A2C with pluggable learners and exploration strategies
- **Healthcare Domain Support**: Pre-built agents, subjects, and PK/PD models for clinical applications
- **Serialization & Checkpointing**: Full support for experiment resumption and backward compatibility
- **Multi-Entity Experiments**: Coordinate multiple agents and subjects with entity registration and statistics
- **TensorBoard Integration**: Automatic logging of learning curves and metrics


Installation
------------

.. code-block:: bash

   pip install ReiL


Quick Start
-----------

**Creating a Simple Agent and Subject:**

.. code-block:: python

    from reil.agents import DeepQLearning
    from reil.subjects import Subject
    from reil.datatypes.feature import Feature, FeatureSet
    from reil.learners.q_learner import QLearner, DeepQModel
    from reil.environments import Single

    # Define a subject (environment)
    class MySubject(Subject):
        def __init__(self):
            super().__init__()
            self.state.add_definition('value', Feature.numerical('value', lower=0, upper=100))
            self.internal_state = 50

        def step(self, action):
            # Update state based on action
            self.internal_state += action['step'].value - 50
            new_state = FeatureSet({'value': Feature.numerical('value', value=self.internal_state, lower=0, upper=100)})
            reward = 1.0 if abs(self.internal_state - 75) < 5 else 0.0
            return new_state, reward

    # Create an agent that learns
    model = DeepQModel(learning_rate=1e-3, hidden_layer_sizes=[64, 32])
    learner = QLearner(model=model)
    agent = DeepQLearning(
        name='my_agent',
        learner=learner,
        exploration_strategy=0.1,
        discount_factor=0.95
    )

    # Create environment and train
    subject = MySubject()
    env = Single(subject=subject, agent=agent)
    for episode in range(100):
        env.step()

    # Save trained agent
    agent.save('my_trained_agent')


Module Overview
---------------

**Core Framework** (reil/):

- ``reilbase.py``: Base class providing persistence, logging, naming
- ``stateful.py``: State management and entity registration layer
- ``datatypes/``: Features, FeatureSets, buffers, interaction protocols

**Learning Algorithms** (reil/learners/):

- ``q_learner.py``: Deep Q-Learning with neural network models
- ``ppo_learner.py``: Proximal Policy Optimization
- ``actor_critic_learner.py``: Actor-Critic methods (A2C)

**Agents & Subjects** (reil/agents/, reil/subjects/):

- Base agent and subject classes
- Agent types: learnable, random, user-controlled
- Environment orchestrators (Sequential, Single, Task)

**Healthcare Domain** (reil/healthcare/):

- ``subjects/warfarin.py``: Warfarin dosing simulation
- ``agents/warfarin_agent.py``: Baseline dosing protocols
- ``mathematical_models/``: PK/PD models (Hamberg, etc.)
- ``patient.py``: Patient data abstractions

**Utilities** (reil/utils/):

- ``yaml_tools.py``: Config loading and dynamic instantiation
- ``exploration_strategies.py``: Epsilon-greedy, decaying exploration
- ``stopping_criteria.py``: Training termination logic
- ``tf_utils.py``: TensorFlow utilities (checkpointing, TensorBoard)


Configuration & YAML
--------------------

ReiL uses YAML configs to define agents, subjects, and experiment parameters without code changes:

.. code-block:: yaml

    # agents.yaml
    dqn_agent: &dqn_agent
      reil.agents.deep_q_learning.DeepQLearning:
        name: my_dqn
        learner:
          reil.learners.q_learner.QLearner:
            model:
              reil.learners.q_learner.DeepQModel:
                learning_rate: 1e-3
                hidden_layer_sizes: [256, 128, 64]
        buffer:
          reil.datatypes.buffers.vanilla_experience_replay.VanillaExperienceReplay:
            buffer_size: 500
            batch_size: 64
        exploration_strategy:
          reil.utils.exploration_strategies.VariableEpsilonGreedy:
            epsilon: "lambda n: 1/(1+n)"
        discount_factor: 0.95

Load and use:

.. code-block:: python

    from reil.utils.yaml_tools import load_config_dict

    config = load_config_dict('agents.yaml')
    agent = config['dqn_agent']  # Ready to train with


Documentation
-------------

**For AI Assistants & Developers:**

- ``copilot-instructions.md``: High-level guidance on abstractions, patterns, and troubleshooting
- ``reil/.instructions.md``: Deep architectural reference and code navigation patterns
- ``docs/ARCHITECTURE.md``: Design principles, extension points, and rationale

**For Learners:**

- ``.skills/YAML_CONFIG_MASTERY.md``: Working with YAML configs, patterns, debugging
- ``.skills/EXPERIMENT_ORCHESTRATION.md``: Setting up and running multi-agent experiments

**Auto-Generated Docs:**

- ``docs/`` (Sphinx): Full API reference

**Example Application:**

See `warfarin_dosing <https://github.com/yourusername/warfarin_dosing>`_ for a complete research project using ReiL.


Contributing
------------

When extending ReiL:

1. Inherit from appropriate base class (ReilBase, Stateful, Agent, Subject, Learner)
2. Follow the Feature/FeatureSet pattern for state/action/reward representation
3. Add docstrings to public methods
4. Include YAML config examples for new entity types
5. Update backward compatibility mapping in ``serialization.py`` if renaming classes

See ``docs/ARCHITECTURE.md`` for extension points and design patterns.


License
-------

See LICENSE file.
