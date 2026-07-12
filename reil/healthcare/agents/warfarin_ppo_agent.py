import copy
from collections.abc import Generator
from typing import Any, Literal

import numpy as np
import tensorflow as tf
from tensorflow import Tensor

from reil.agents.agent import BaseAgent, TrainingData
from reil.agents.ppo_agent import PPOAgent, PPOLearner
from reil.datatypes.buffers.buffer import Buffer
from reil.datatypes import History, Observation
from reil.datatypes.feature import FeatureGeneratorType, FeatureSet
from reil.utils.action_dist_modifier import ActionModifier
from reil.utils.metrics import ActionMetric, INRMetric, PTTRMetric
from reil.utils.tf_utils import MeanMetric


class PPO4WarfarinAgent(PPOAgent):
    def __init__(
            self, learner: PPOLearner,
            buffer: Buffer[FeatureSet, tuple[tuple[int, ...], float, float]],
            reward_clip: tuple[float | None, float | None] = ...,
            gae_lambda: float = 1, momentum_coef: float = 0.,
            momentum_mode: Literal['most recent', 'carry'] | None = None,
            action_modifiers: list[ActionModifier] | None = None,
            **kwargs: Any):
        super().__init__(learner, buffer, reward_clip, gae_lambda, **kwargs)
        # PTTR_h / INR_h histograms dropped — fully reconstructible from
        # trajectories/ post-hoc, and the per-iter histogram bucketing is
        # expensive under --slots N contention. Scalar PTTR / INR retained.
        self._metrics['PTTR'] = PTTRMetric('PTTR', mode='scalar')
        self._metrics['INR'] = INRMetric('INR', mode='scalar')
        self._metrics['dose'] = ActionMetric('dose', 0)
        self._metrics['duration'] = ActionMetric('duration', 1)

        self._previous_action = [
            tf.zeros(x) for x in self._learner._model._action_per_head]
        self._momentum_coef = momentum_coef
        self._carry = momentum_mode == 'carry'
        self._action_modifiers = action_modifiers or []
        # None entries are valid placeholders for heads with no modifier (e.g.
        # a duration-only comb passes [None, comb]); act() skips them too.
        for modifier in self._action_modifiers:
            if modifier is None:
                continue
            self._metrics[f'modifier_{modifier.name}'] = MeanMetric(
                f'modifier_{modifier.name}_scale', dtype=tf.float32)

        if self._summary_writer:
            self._summary_writer.set_data_types({
                'dose': 'histogram', 'duration': 'histogram'
            })
            # 21 = action_count for the Ch.2 dose-percent-change head.
            # Exact per-action counts beat the default 30-bin smoothing.
            self._summary_writer.set_buckets({'dose': 21})

    def _update_metrics(self, **kwargs: Any) -> None:
        super()._update_metrics(**kwargs)

        state_list = kwargs.get('state_list')
        if state_list:
            self._metrics['PTTR'].update_state(state_list)
            self._metrics['INR'].update_state(state_list)

        action_indices = kwargs.get('action_indices')
        if action_indices:
            self._metrics['dose'].update_state(action_indices)
            try:
                self._metrics['duration'].update_state(action_indices)
            except IndexError:
                pass

    @staticmethod
    def _sample_head(logits_row, mask_idx: list[int], training_mode: bool) -> int:
        '''Mask a head's logits to the permissible actions and sample (train:
        softmax-categorical; eval: argmax), returning the FULL-space action
        index. NumPy — no per-decision eager TF ops (see the act() leak fix).'''
        masked = np.asarray(logits_row)[mask_idx]
        if training_mode:
            e = np.exp(masked - masked.max())
            p = e / e.sum()
            k = int(np.random.choice(len(p), p=p))
        else:
            k = int(np.argmax(masked))
        return mask_idx[k]

    def _act_conditional_tandem(
            self, state: FeatureSet, actions: FeatureGeneratorType,
            model, training_mode: bool) -> FeatureSet:
        '''Two-stage rollout for PPOTandemConditionalModel: sample the dose,
        then sample the duration conditioned on the prescribed dose. Masks come
        from the same [dose, duration] mask vector as the non-conditional
        tandem (head 0 = dose, head 1 = duration).'''
        from reil.utils.tf_utils import TF2UtilsMixin
        state_tensor = TF2UtilsMixin.convert_to_tensor((state,))
        mask = list(actions.send('return mask_vector'))
        mask_index = [[i for i, j in enumerate(m) if j] for m in mask]

        dose_logits = model.act_dose_logits(state_tensor)[0]
        if training_mode:
            dose_logits = self._apply_head_modifier(0, dose_logits)
        dose_idx = self._sample_head(dose_logits, mask_index[0], training_mode)

        dur_logits = model.act_duration_logits(
            state_tensor, tf.constant([dose_idx], dtype=tf.int32))[0]
        if training_mode:
            dur_logits = self._apply_head_modifier(1, dur_logits)
        dur_idx = self._sample_head(dur_logits, mask_index[1], training_mode)

        return actions.send(f'lookup {[dose_idx, dur_idx]}')

    def _apply_head_modifier(self, head: int, logits_row):
        '''Apply the action modifier for `head` (e.g. the duration comb) to a
        1-D logits row, before sampling. The two-stage conditional-tandem
        rollout samples inside `_act_conditional_tandem` and returns before the
        standard act() modifier loop, so modifiers must be applied here to take
        effect on the sampled action. No-op when no modifier is set for `head`.'''
        if head >= len(self._action_modifiers):
            return logits_row
        modifier = self._action_modifiers[head]
        if modifier is None:
            return logits_row
        out = modifier(tf.expand_dims(logits_row, axis=0))[0]
        self._metrics[f'modifier_{modifier.name}'].update_state(
            modifier._scale_fn.last_call)
        return out

    def act(
            self, state: FeatureSet, subject_id: int,
            actions: FeatureGeneratorType, iteration: int = 0) -> FeatureSet:
        if subject_id not in self._entity_list:
            raise ValueError(f'Subject with ID={subject_id} not found.')

        training_mode = self._training_trigger != 'none'
        model = getattr(self._learner, '_model', None)
        # Conditional tandem (Axis A): the duration head conditions on the
        # PRESCRIBED dose, so we must sample the dose FIRST, then the duration
        # given that dose. Two-stage path; the standard single-forward path
        # below is unchanged for every other model.
        if getattr(model, 'act_dose_logits', None) is not None:
            return self._act_conditional_tandem(
                state, actions, model, training_mode)
        # Fast path: if the model exposes a tf.function-decorated
        # `actor_logits` (PPOModel + subclasses), use it. Saves the
        # critic forward (not needed for act()) and avoids the per-op
        # eager-dispatch overhead that dominated the 2026-06-08 profile
        # (918K eager-execute calls per chunk under the legacy path).
        actor_logits_fn = getattr(model, 'actor_logits', None)
        if actor_logits_fn is not None:
            from reil.utils.tf_utils import TF2UtilsMixin
            state_tensor = TF2UtilsMixin.convert_to_tensor((state,))
            raw_logits = actor_logits_fn(state_tensor)
        else:
            # Legacy path for non-PPO models (LookupTable, Dense, etc.).
            raw_logits = self._learner.predict(
                (state,), training=training_mode)[0]
        if isinstance(raw_logits, (list, tuple)):
            logits: list[Tensor] = list(raw_logits)  # type: ignore
        else:
            logits = [raw_logits]  # type: ignore

        # Single-head models may return rank-1 tensors (n_actions,).
        # Normalize to rank-2 (1, n_actions) to keep masking logic uniform.
        logits = [
            tf.expand_dims(lo, axis=0) if lo.shape.rank == 1 else lo
            for lo in logits
        ]
        if training_mode:
            temp = logits
            for i, x in enumerate(self._previous_action):
                logits[i] += tf.multiply(self._momentum_coef, x)  # type: ignore
            self._previous_action = logits if self._carry else temp

            for i, modifier in enumerate(self._action_modifiers):
                if modifier is not None:
                    logits[i] = modifier(logits[i])
                    self._metrics[f'modifier_{modifier.name}'].update_state(
                        modifier._scale_fn.last_call)

        mask = list(actions.send('return mask_vector'))
        # The mask cardinality must equal the model head's width: one logit
        # per possible action, one mask bit per possible action. A mismatch
        # means the actions generator was built against a different action
        # space than the model — almost always config drift between training
        # and the current call (e.g., a pickled model loaded against a wider
        # action set, or init_action_name vs main_action_name disagreement).
        # We raise here rather than silently clip+fallback, because a silent
        # fallback violates the environment's mask contract and corrupts
        # downstream PTTR / trajectory results without leaving an audit trail.
        for i, (lo, m_vec) in enumerate(zip(logits, mask)):
            upper = lo.shape[1]
            if upper is None:
                continue
            if len(m_vec) != upper:
                raise ValueError(
                    f"Action mask cardinality {len(m_vec)} for head {i} "
                    f"doesn't match model head width {upper}. "
                    f"Likely cause: the model's action_per_head doesn't "
                    f"match the current actions generator. Check that the "
                    f"model was trained with the same action space as the "
                    f"current environment (often init_action_name vs "
                    f"main_action_name, or a stale pickled agent)."
                )
        mask_index = [
            [i for i, j in enumerate(m) if j]
            for m in mask
        ]

        # Mask + sample per head in NumPy. This deliberately does NOT use
        # per-decision `tf.gather` + `tf.random.categorical`: under Ch.3's
        # variable dosing interval the policy converges toward short durations,
        # so decisions-per-patient explodes (~10 early -> ~80+ late) and those
        # two eager ops leaked ~110 KB of TensorFlow C++ memory *per decision*
        # in the long training process (not reproducible standalone; RSS grew
        # ~4.5 MB/patient and OOM'd a 31 GB box on a 5k-patient S1 run,
        # root-caused 2026-07-04). The forward stays on the traced
        # `actor_logits` path; only the masking + categorical sampling move to
        # NumPy. Sampling from `softmax(masked_logits)` is exactly the
        # distribution `tf.random.categorical(logits=masked_logits)` draws, and
        # np.random is seeded by `reil.set_reil_random_seed`, so training is
        # still reproducible (the RNG stream differs from the old tf.random one,
        # so pre-fix runs won't reproduce bit-for-bit — acceptable for new runs).
        permissible_action_index = []
        for lo, mi in zip(logits, mask_index):
            row = np.asarray(lo)[0][mi]  # gather permissible logits (1-D)
            if training_mode:
                e = np.exp(row - row.max())
                p = e / e.sum()
                permissible_action_index.append(
                    int(np.random.choice(len(p), p=p)))
            else:
                permissible_action_index.append(int(np.argmax(row)))

        if len(permissible_action_index) == 1:
            # In the implementation of feature.byindex(), if index is one dimensional
            # it excludes masked values. Hence, the following:
            action_index = permissible_action_index[0]
        else:
            action_index = [
                mask_index[i][permissible_action_index[i]]
                for i in range(len(permissible_action_index))
            ]

        action: FeatureSet = actions.send(f'lookup {action_index}')

        return action

    def reset(self) -> None:
        self._previous_action = [
            tf.zeros(x) for x in self._learner._model._action_per_head]
        return super().reset()


class PPO4Warfarin2PhaseAgent(PPO4WarfarinAgent):
    def __init__(
        self, init_agent: BaseAgent, switch_day: int,
        init_state_comps: tuple[str, ...], main_state_comps: tuple[str, ...],
        learner: PPOLearner,
        buffer: Buffer[FeatureSet, tuple[tuple[int, ...], float, float]],
        reward_clip: tuple[float | None, float | None] = ...,
        gae_lambda: float = 1, momentum_coef: float = 0,
        momentum_mode: Literal['most recent', 'carry'] | None = None,
        **kwargs: Any
    ):
        super().__init__(
            learner, buffer, reward_clip,
            gae_lambda, momentum_coef, momentum_mode, **kwargs)
        self._init_agent = init_agent
        self._switch_day = switch_day
        self._init_state_comps = init_state_comps
        self._main_state_comps = main_state_comps

    def act(
            self, state: FeatureSet, subject_id: int,
            actions: FeatureGeneratorType, iteration: int = 0) -> FeatureSet:
        val = state.value
        if val['day'] < self._switch_day:  # type: ignore
            for f in set(val.keys()).difference(self._init_state_comps):
                state.pop(f)

            action = self._init_agent.act(state, subject_id, actions, iteration)
            return action

        for f in set(val.keys()).difference(self._main_state_comps):
            state.pop(f)

        return super().act(state, subject_id, actions, iteration)

    def _prepare_training(self, history: History) -> TrainingData[FeatureSet, int]:
        # days = [
        #     90 - t for t in self.discounted_cum_sum([
        #         h.state['duration_history'].value[-1]  # type: ignore
        #         for h in history if h.state is not None], 1)
        # ][1:]
        # temp = [h for h, d in zip(history, days) if d > self._switch_day]
        temp = [h for h in history if h.state is not None and 'day' not in h.state.value]

        return super()._prepare_training(temp)


class ActionSplitter:
    def __init__(self, action_gen: FeatureGeneratorType, index: int = 0) -> None:
        self._action_gen = action_gen
        self._index = index

    def send(self, query: str | None):
        if query is None:
            self._action_gen.send(None)  # type: ignore
            return

        if query.startswith('return'):
            if 'split' not in query:
                query += ' split'
            result = list(self._action_gen.send(query))
            return [result[self._index]]

        if query.startswith('lookup'):
            index = int(query.split()[1])
            features: list[FeatureSet] = list(self._action_gen.send('return feature exclusive split'))
            i, f = -1, None
            for i, f in zip(range(index + 1), features[self._index]):
                pass
            if i == index:
                return f
            raise IndexError('index not found.')
        else:
            raise RuntimeError('query not supported by ActionSplitter.')


class PPO4Warfarin2PartAgent(BaseAgent):
    def __init__(
            self, dose_agent: PPO4WarfarinAgent | PPO4Warfarin2PhaseAgent,
            duration_agent: PPO4WarfarinAgent | PPO4Warfarin2PhaseAgent,
            training_switch: tuple[tuple[str, int], ...] | None = None,
            dose_first: bool = True):
        super().__init__()
        self._dose_agent = dose_agent
        self._duration_agent = duration_agent

        if self._dose_agent._training_trigger != self._duration_agent._training_trigger:
            raise ValueError('training_trigger must be the same for both agents.')
        self._training_trigger = self._dose_agent._training_trigger

        self._dose_first = dose_first
        self._training_switch = training_switch or (('all', 10000),)
        self._training_counter: int = 0
        self._current_training_index = 0

    def register(self, entity_name: str, _id: int | None = None) -> int:
        self._dose_agent.register(entity_name=entity_name, _id=_id)
        return self._duration_agent.register(entity_name=entity_name, _id=_id)

    def act(
            self, state: FeatureSet, subject_id: int,
            actions: FeatureGeneratorType, iteration: int = 0) -> FeatureSet:
        state_copy = copy.deepcopy(state)
        if self._dose_first:
            dose = self._dose_agent.act(
                state, subject_id,
                ActionSplitter(actions, 0), iteration
            )
            if 'duration' in dose.value:
                del dose['duration']

            duration = self._duration_agent.act(
                state_copy + dose, subject_id,
                ActionSplitter(actions, 1), iteration
            )
            if 'dose' in duration.value:
                del duration['dose']
        else:
            duration = self._duration_agent.act(
                state, subject_id,
                ActionSplitter(actions, 1), iteration
            )
            if 'dose' in duration.value:
                del duration['dose']

            dose = self._dose_agent.act(
                state_copy + duration, subject_id,
                ActionSplitter(actions, 0), iteration
            )
            if 'duration' in dose.value:
                del dose['duration']

        return dose + duration

    @staticmethod
    def _split_history(
            history: History, dose_first: bool = True) -> tuple[History, History]:
        dose_history, duration_history = History(), History()
        for h in history:
            o_dose = Observation(
                None, h.possible_actions, None, None, h.lookahead, h.reward)
            o_duration = Observation(
                None, h.possible_actions, None, None, h.lookahead, h.reward)
            action = h.action
            if action is not None:
                dose_part_name = [a for a in action.value if 'dose' in a][0]
                o_dose.action = FeatureSet(action[dose_part_name])
                o_duration.action = FeatureSet(
                    a for a in action
                    if dose_part_name not in a.name
                )
                if h.action_taken is not None:
                    o_dose.action_taken = FeatureSet(h.action_taken[dose_part_name])
                    o_duration.action_taken = FeatureSet(
                        a for a in h.action_taken
                        if dose_part_name not in a.name
                    )
                if h.state is not None:
                    if dose_first:
                        o_dose.state = h.state
                        o_duration.state = h.state + (o_dose.action_taken or o_dose.action)
                    else:
                        o_duration.state = h.state
                        o_dose.state = h.state + (o_duration.action_taken or o_duration.action)

            dose_history.append(o_dose)
            duration_history.append(o_duration)

        return dose_history, duration_history

    def learn(self, history: History) -> dict[str, float]:
        '''
        Learn using history.

        Arguments
        ---------
        subject_id:
            the ID of the `subject` whose history is being used for learning.

        next_state:
            The new `state` of the `subject` after taking `agent`'s action.
            Some methods
        '''
        if history is not None:
            dose_history, duration_history = self._split_history(
                history, self._dose_first)

            key = self._training_switch[self._current_training_index][0]
            iteration = max(
                self._dose_agent._learner._iteration,
                self._duration_agent._learner._iteration
            )
            metrics = {}
            metrics_temp = {}
            if key in ('dose', 'all'):
                metrics = self._dose_agent.learn(dose_history)
                # metrics = {
                #     f'dose_{name}': value
                #     for name, value in metrics.items()
                # }

                if self._dose_agent._summary_writer:
                    self._dose_agent._summary_writer.write(metrics, iteration)

            if key in ('duration', 'all'):
                metrics_temp = self._duration_agent.learn(duration_history)
                if not metrics:
                    metrics = {
                        name: m
                        for name, m in metrics_temp.items()
                        if name in ('PTTR_h', 'INR_h', 'PTTR', 'INR', 'duration')
                    }
                    if self._dose_agent._summary_writer:
                        self._dose_agent._summary_writer.write(metrics, iteration)
                metrics_temp = {
                    name: m
                    for name, m in metrics_temp.items()
                    if name not in ('PTTR_h', 'INR_h', 'PTTR', 'INR')
                }

                # metrics_temp = {
                #     f'duration_{name}': value
                #     for name, value in metrics.items()
                # }

                if self._duration_agent._summary_writer:
                    self._duration_agent._summary_writer.write(metrics_temp, iteration)

            metrics.update(metrics_temp)

            if metrics:  # training has really happened!
                self._training_counter += 1
                if self._training_counter >= self._training_switch[self._current_training_index][1]:
                    self._training_counter = 0
                    self._current_training_index += 1
                    if self._current_training_index >= len(self._training_switch):
                        self._current_training_index = 0

        return metrics

    def observe(  # noqa: C901
            self, subject_id: int, stat_name: str | None,
    ) -> Generator[FeatureSet | None, dict[str, Any], None]:
        '''
        Create a generator to interact with the subject (`subject_id`).
        Extends `BaseAgent.observe`.

        This method creates a generator for `subject_id` that
        receives `state`, yields `action` and receives `reward`
        until it is closed. When `.close()` is called on the generator,
        `statistics` are calculated.

        Arguments
        ---------
        subject_id:
            the ID of the `subject` on which action happened.

        stat_name:
            The name of the `statistic` that should be computed at the end of
            each trajectory.

        Raises
        ------
        ValueError
            Subject with `subject_id` not found.
        '''
        if (subject_id not in self._dose_agent._entity_list) or (subject_id not in self._duration_agent._entity_list):
            raise ValueError(f'Subject with ID={subject_id} not found.')

        # trigger = self._training_trigger
        # learn_on_state = trigger == 'state'
        # learn_on_action = trigger == 'action'
        # learn_on_reward = trigger == 'reward'
        # learn_on_termination = trigger == 'termination'

        history: History = []
        new_observation = None
        # Per-trajectory flag: True iff `learn()` actually produced metrics
        # this trajectory. Gates the end-of-trajectory TB write so test
        # passes (trigger='none') don't re-emit the previous training step's
        # `_computed_metrics` at a frozen learner iteration — which otherwise
        # shows up as a spurious jump in TB curves at the train→test
        # boundary. `n_actions_alive` and other in-`learn()` writes
        # (warfarin_ppo_agent.learn lines 366/377/390) are unaffected.
        learned_this_trajectory = False
        while True:
            try:
                new_observation = Observation()
                temp: dict[str, Any] = yield
                state: FeatureSet = temp['state']
                possible_actions: FeatureGeneratorType = temp['possible_actions']
                iteration: int = temp['iteration']

                new_observation.state = state
                new_observation.possible_actions = possible_actions
                # if learn_on_state:
                #     self._computed_metrics.update(
                #         self.learn([history[-1], new_observation]))

                if possible_actions is not None:
                    new_observation.action = self.act(
                        state=state, subject_id=subject_id,
                        actions=possible_actions, iteration=iteration)

                    temp = yield new_observation.action

                    new_observation.action_taken = temp['action_taken']
                    new_observation.lookahead = temp.get('lookahead')

                    # if learn_on_action:
                    #     self._computed_metrics.update(
                    #         self.learn([history[-1], new_observation]))

                    new_observation.reward = (yield None)['reward']

                    history.append(new_observation)

                    # if learn_on_reward:
                    #     self._computed_metrics.update(self.learn(history[-2:]))
                else:  # No actions to take, so skip the reward.
                    yield

            except GeneratorExit:
                if new_observation is None:
                    new_observation = Observation()
                if new_observation.reward is None:  # terminated early!
                    history.append(new_observation)

                # if learn_on_termination:
                    # self._computed_metrics = self.learn(history)
                if self._dose_agent._training_trigger == 'termination':
                    m = self.learn(history)
                    self._computed_metrics.update(m)
                    learned_this_trajectory = (
                        learned_this_trajectory or bool(m))

                if self._summary_writer and learned_this_trajectory:
                    self._summary_writer.write(
                        self._computed_metrics, self._learner._iteration)
                    # Flush immediately for visibility
                    tf_writer = getattr(self._summary_writer, '_summary_writer', None)
                    if tf_writer is not None:
                        tf_writer.flush()

                if stat_name is not None:
                    self.statistic.append(stat_name, subject_id)

                self.reset()

                return

    def get_parameters(self) -> Any:
        return {
            'dose_agent': self._dose_agent.get_parameters(),
            'duration_agent': self._duration_agent.get_parameters(),
            'dose_first': self._dose_first
        }

    def set_parameters(self, parameters: Any):
        self._dose_agent.set_parameters(parameters['dose_agent'])
        self._duration_agent.set_parameters(parameters['duration_agent'])
        self._dose_first = parameters.get('dose_first', True)


class PPO4WarfarinSeparateAgent(PPO4Warfarin2PartAgent):
    @staticmethod
    def _split_history(
            history: History, dose_first: bool = True) -> tuple[History, History]:
        dose_history, duration_history = PPO4Warfarin2PartAgent._split_history(
            history, dose_first)
        for h in duration_history:
            if h.state is not None:
                state_val = h.state.value
                tau: int = state_val['duration_history'][-1]
                inr: float = state_val['INR_history'][-2]
                d = abs(inr - 2.5)
                h.reward = -(
                    tau * d * 0.3 if d > 0.5 else
                    1. * (28 - tau) * (0.5 - d))

        return dose_history, duration_history


class PPO4WarfarinSeparateSimpleRewardAgent(PPO4Warfarin2PartAgent):
    @staticmethod
    def _split_history(
            history: History, dose_first: bool = True) -> tuple[History, History]:
        dose_history, duration_history = PPO4Warfarin2PartAgent._split_history(
            history, dose_first)
        for h in duration_history:
            if h.state is not None:
                state_val = h.state.value
                tau: int = state_val['duration_history'][-1]
                inr: float = state_val['INR_history'][-1]
                d = abs(inr - 2.5)
                if d > 0.5:
                    h.reward = -1. if tau > 1 else 0.
                else:
                    h.reward = (tau - 7) / 28

        return dose_history, duration_history


class PPO4WarfarinTandemPerHeadAgent(PPO4Warfarin2PhaseAgent):
    '''Two-phase tandem agent that trains each head on its OWN reward (JA-2).

    Extends PPO4Warfarin2PhaseAgent so it keeps the exact NEWDOSE/RAMP training
    setup (PGAA init for day < switch_day, then the conditional tandem policy;
    init-phase observations dropped in `_prepare_training`). The ONLY change is
    the training-data assembly: the dose head uses the primary reward (control)
    while the duration head uses `Observation.reward_2` — the burden+safety
    ramp reward the environment fills when the protocol sets `reward_name_2`.
    Two per-head returns/advantages are computed against the tandem critic's two
    value outputs and packed as `[·, head_count]` tuples, so a `PPOTandemModel`
    built with `per_head_advantage=True` routes each head to its own advantage
    column via `train_actor_per_head`/`train_critic_per_head`. This removes the
    shared-advantage contamination (dose control variance swamping the duration
    signal) that pinned the duration head at tau=1.

    Head order is (dose, duration) == action_per_head order == critic column
    order. `act()` and everything else are inherited unchanged.
    '''

    def _prepare_training(
            self, history: History) -> TrainingData[FeatureSet, int]:
        # Drop the init (PGAA) phase exactly as PPO4Warfarin2PhaseAgent does:
        # only main-phase observations (state has no 'day' feature) are learned.
        history = [
            h for h in history
            if h.state is not None and 'day' not in h.state.value]

        discount_factor = self._discount_factor
        active_history = self.get_active_history(history)

        # add a trailing 0 so `deltas` (len N) line up (mirrors PPOAgent).
        rewards_dose = self.extract_reward(
            active_history, *self._reward_clip) + [0.0]
        rewards_dur = [
            float(h.reward_2) if h.reward_2 is not None else 0.0
            for h in active_history] + [0.0]

        dis_dose = self.discounted_cum_sum(rewards_dose, discount_factor)
        dis_dur = self.discounted_cum_sum(rewards_dur, discount_factor)

        state_list: tuple[FeatureSet, ...] = tuple(
            h.state for h in active_history)  # type: ignore
        y, values = self._learner.predict(state_list)
        values = np.asarray(values, dtype=np.float32)  # [N, head_count]
        if values.ndim == 1:  # scalar critic — per_head_advantage not set
            raise RuntimeError(
                'PPO4WarfarinTandemPerHeadAgent needs a per-head critic; build '
                'the tandem model with per_head_advantage=True.')
        # terminal bootstrap row of zeros (mirrors the scalar-value path).
        values = np.vstack(
            [values, np.zeros((1, values.shape[1]), dtype=np.float32)])

        action_indices = tuple(
            list((h.action_taken or h.action).index.values())  # type: ignore
            for h in active_history)

        gl = discount_factor * self._gae_lambda
        deltas_dose = (
            np.asarray(rewards_dose[:-1], dtype=np.float32)
            + discount_factor * values[1:, 0] - values[:-1, 0])
        adv_dose = self.discounted_cum_sum(list(deltas_dose), gl)
        deltas_dur = (
            np.asarray(rewards_dur[:-1], dtype=np.float32)
            + discount_factor * values[1:, 1] - values[:-1, 1])
        adv_dur = self.discounted_cum_sum(list(deltas_dur), gl)

        self._buffer.add_iter({
            'state': h.state,
            'y_r_a': (
                tuple((h.action_taken or h.action).index.values()),  # type: ignore
                (dr_d, dr_u), (a_d, a_u))
        } for h, dr_d, dr_u, a_d, a_u in zip(
            active_history, dis_dose, dis_dur, adv_dose, adv_dur))

        temp = self._buffer.pick()

        # advantage metric reports the DURATION head — the one under study.
        self._update_metrics(
            rewards=rewards_dose, dis_reward=dis_dose, state_list=state_list,
            y=y, values=values, action_indices=action_indices,
            deltas=deltas_dose, advantage=adv_dur)

        return temp['state'], temp['y_r_a'], {}  # type: ignore
