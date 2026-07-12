import unittest

import numpy as np
import tensorflow as tf

from reil.learners.ppo_learner_tandem import PPOTandemModel
from reil.utils.tf_utils import GradientGate

# Tiny tandem: 4 state features -> dose head (3) + duration head (2).
INPUT_DIM = 4
HEADS = (3, 2)
BATCH = 6


def _build_model(coupling_gradient='full', training_schedule=None,
                 head_loss_weights=None, trunk=None,
                 regularizer_coef=0.0) -> PPOTandemModel:
    layer_sizes = {'dose': (8,), 'duration': (6,)}
    if trunk:
        layer_sizes = {'trunk': trunk, **layer_sizes}
    return PPOTandemModel(
        input_shape=(INPUT_DIM,),
        action_per_head=HEADS,
        actor_learning_rate=1e-2,
        critic_learning_rate=1e-3,
        actor_layer_sizes=layer_sizes,
        critic_layer_sizes=(8,),
        actor_train_iterations=2,
        critic_train_iterations=1,
        target_kl=100.0,  # keep the KL early-stop out of the way
        training_schedule=training_schedule,
        coupling_gradient=coupling_gradient,
        head_loss_weights=head_loss_weights,
        regularizer_coef=regularizer_coef,
        clip_ratio=0.2,
    )


def _batch():
    rng = np.random.RandomState(0)
    x = tf.constant(rng.randn(BATCH, INPUT_DIM).astype(np.float32))
    actions = tf.constant(
        np.stack([rng.randint(0, n, BATCH) for n in HEADS], axis=1)
        .astype(np.int32))
    returns = tf.constant(rng.randn(BATCH).astype(np.float32))
    advantage = tf.constant(rng.randn(BATCH).astype(np.float32))
    return x, actions, returns, advantage


def _section_vars(model, section):
    '''Trainable weights of a section = its hidden stack + its Dense head.'''
    return [
        w for layer in model.actor.layers
        if f'_{section}_' in layer.name or layer.name == f'actor_{section}_output'
        for w in layer.trainable_weights]


def _duration_loss_grads(model, variables):
    '''Gradient of a duration-head-only scalar w.r.t. `variables`.'''
    x, *_ = _batch()
    with tf.GradientTape() as tape:
        logits = model.actor(x)
        duration_scalar = tf.reduce_sum(tf.square(logits[1]))
    return tape.gradient(duration_scalar, variables)


def _grads_nonzero(grads):
    return any(
        g is not None and float(tf.reduce_max(tf.abs(g))) > 0.0
        for g in grads)


def _snapshot(variables):
    return [v.numpy().copy() for v in variables]


def _max_delta(before, variables):
    return max(
        float(np.max(np.abs(b - v.numpy())))
        for b, v in zip(before, variables))


class TestCouplingGradient(unittest.TestCase):
    '''coupling_gradient routes ONLY the section-to-section coupling.'''

    def test_full_couples_duration_loss_to_dose_section(self):
        model = _build_model('full')
        self.assertTrue(_grads_nonzero(
            _duration_loss_grads(model, _section_vars(model, 'dose'))))

    def test_blocked_severs_the_coupling(self):
        model = _build_model('blocked')
        grads = _duration_loss_grads(model, _section_vars(model, 'dose'))
        for g in grads:
            if g is not None:
                self.assertEqual(float(tf.reduce_max(tf.abs(g))), 0.0)

    def test_gated_closed_equals_blocked_and_open_couples(self):
        model = _build_model('gated')
        dose_vars = _section_vars(model, 'dose')
        for g in _duration_loss_grads(model, dose_vars):
            if g is not None:
                self.assertEqual(float(tf.reduce_max(tf.abs(g))), 0.0)
        gates = [layer for layer in model.actor.layers
                 if isinstance(layer, GradientGate)]
        self.assertEqual(len(gates), 1)
        gates[0].gate.assign(1.0)
        self.assertTrue(_grads_nonzero(
            _duration_loss_grads(model, dose_vars)))

    def test_retired_mode_names_rejected(self):
        with self.assertRaises(ValueError):
            _build_model('shared')


class TestSharedTrunk(unittest.TestCase):
    '''Optional 'trunk' entry: a shared latent trained by EVERY head.'''

    def test_trunk_layers_exist_and_feed_sections(self):
        model = _build_model('blocked', trunk=(5,))
        trunk_layers = [layer for layer in model.actor.layers
                        if '_trunk_' in layer.name]
        self.assertEqual(len(trunk_layers), 1)

    def test_trunk_gets_duration_gradient_even_when_blocked(self):
        model = _build_model('blocked', trunk=(5,))
        trunk_vars = [w for layer in model.actor.layers
                      if '_trunk_' in layer.name
                      for w in layer.trainable_weights]
        # The vertical path (duration stack -> z -> trunk) is never severed…
        self.assertTrue(_grads_nonzero(
            _duration_loss_grads(model, trunk_vars)))
        # …while the horizontal coupling into the dose section still is.
        for g in _duration_loss_grads(model, _section_vars(model, 'dose')):
            if g is not None:
                self.assertEqual(float(tf.reduce_max(tf.abs(g))), 0.0)


class TestHeadLossWeights(unittest.TestCase):
    '''(0, 1) + 'full' = downstream-only (dissertation §3.2.2 "shared");
    (0, 1) + 'blocked' leaves the dose section completely untrained.'''

    def _train_once(self, model):
        x, actions, returns, advantage = _batch()
        model.train_step((x, (actions, returns, advantage)))

    def test_zero_weight_blocked_freezes_dose_section(self):
        model = _build_model('blocked', head_loss_weights=(0.0, 1.0))
        dose_vars = _section_vars(model, 'dose')
        before = _snapshot(dose_vars)
        self._train_once(model)
        self.assertEqual(_max_delta(before, dose_vars), 0.0)

    def test_zero_weight_full_still_trains_dose_via_coupling(self):
        model = _build_model('full', head_loss_weights=(0.0, 1.0))
        dose_vars = _section_vars(model, 'dose')
        before = _snapshot(dose_vars)
        self._train_once(model)
        self.assertGreater(_max_delta(before, dose_vars), 0.0)

    def test_weight_length_validated(self):
        with self.assertRaises(ValueError):
            _build_model('full', head_loss_weights=(1.0,))


class TestTrainingSchedule(unittest.TestCase):
    '''Alternating freeze via gradient masks ('gated' coupling).

    {'heads': 2, 'trunk': 2}: step 1 trains under 'heads' (trunk frozen
    initially), step 2 flips to 'trunk' at the top of train_step. Without a
    real trunk entry, 'trunk' = all non-head layers (the de-facto trunk).
    '''

    def _run(self):
        model = _build_model(
            'gated', training_schedule={'heads': 2, 'trunk': 2})
        x, actions, returns, advantage = _batch()
        data = (x, (actions, returns, advantage))
        head_vars = [
            w for layer in model.actor.layers
            if '_output' in layer.name for w in layer.trainable_weights]
        trunk_vars = [
            w for layer in model.actor.layers
            if '_output' not in layer.name for w in layer.trainable_weights]
        return model, data, head_vars, trunk_vars

    def test_heads_phase_moves_heads_only(self):
        model, data, head_vars, trunk_vars = self._run()
        heads_before = _snapshot(head_vars)
        trunk_before = _snapshot(trunk_vars)
        model.train_step(data)  # step 1: 'heads' phase
        self.assertGreater(_max_delta(heads_before, head_vars), 0.0)
        # Never-trained group has zero Adam state -> updates are EXACTLY 0.
        self.assertEqual(_max_delta(trunk_before, trunk_vars), 0.0)

    def test_trunk_phase_moves_trunk_and_opens_gate(self):
        model, data, head_vars, trunk_vars = self._run()
        model.train_step(data)  # step 1: 'heads'
        heads_before = _snapshot(head_vars)
        trunk_before = _snapshot(trunk_vars)
        model.train_step(data)  # step 2: flips to 'trunk', trains under it
        gate = [layer for layer in model.actor.layers
                if isinstance(layer, GradientGate)][0]
        self.assertEqual(float(gate.gate.numpy()), 1.0)
        self.assertGreater(_max_delta(trunk_before, trunk_vars), 0.0)
        # Heads are frozen; only the decaying Adam momentum tail may move
        # them, which must be far smaller than the trunk's real update.
        self.assertLess(
            _max_delta(heads_before, head_vars),
            _max_delta(trunk_before, trunk_vars))

    def test_real_trunk_phase_targets_trunk_layers_only(self):
        model = _build_model(
            'gated', training_schedule={'heads': 2, 'trunk': 2}, trunk=(5,))
        x, actions, returns, advantage = _batch()
        data = (x, (actions, returns, advantage))
        trunk_vars = [w for layer in model.actor.layers
                      if '_trunk_' in layer.name
                      for w in layer.trainable_weights]
        # Hidden stacks only: '_dose_' would also match 'actor_dose_output'
        # (the momentum tail moves frozen heads slightly), so exclude heads.
        stack_vars = [w for layer in model.actor.layers
                      if ('_dose_' in layer.name or '_duration_' in layer.name)
                      and '_output' not in layer.name
                      for w in layer.trainable_weights]
        model.train_step(data)  # 'heads'
        stacks_before = _snapshot(stack_vars)
        trunk_before = _snapshot(trunk_vars)
        model.train_step(data)  # 'trunk' phase: ONLY the real trunk trains
        self.assertGreater(_max_delta(trunk_before, trunk_vars), 0.0)
        self.assertEqual(_max_delta(stacks_before, stack_vars), 0.0)


class TestDiagnostics(unittest.TestCase):
    def test_per_neuron_l2_vector_covers_all_heads(self):
        model = _build_model('full', trunk=(5,))
        vec = model.per_neuron_l2_vector().numpy()
        self.assertEqual(vec.shape, (sum(HEADS),))
        self.assertTrue(np.all(np.isfinite(vec)))

    def _reg_grads(self, model):
        dose_head = model.actor.get_layer('actor_dose_output')
        dur_head = model.actor.get_layer('actor_duration_output')
        with tf.GradientTape() as tape:
            reg = model._compute_regularizer_loss()
        g_dose = tape.gradient(reg, dose_head.trainable_weights)
        with tf.GradientTape() as tape2:
            reg2 = model._compute_regularizer_loss()
        g_dur = tape2.gradient(reg2, dur_head.trainable_weights)
        return g_dose, g_dur

    @staticmethod
    def _has_grad(gs):
        return any(g is not None and float(tf.reduce_max(tf.abs(g))) > 0.0
                   for g in gs)

    def test_regularizer_scalar_targets_dose_head_only(self):
        # A SCALAR regularizer_coef regularizes the DOSE head only (Paper-2
        # forging default, back-compat): dose gradient present, duration None.
        g_dose, g_dur = self._reg_grads(_build_model('full', regularizer_coef=0.1))
        self.assertTrue(self._has_grad(g_dose))
        self.assertTrue(all(g is None for g in g_dur))

    def test_regularizer_per_head_duration_only(self):
        # A per-head [0, coef] regularizes the DURATION head only (Paper-3):
        # duration gradient present, dose None. This is "act on each head
        # separately".
        g_dose, g_dur = self._reg_grads(
            _build_model('full', regularizer_coef=[0.0, 0.5]))
        self.assertTrue(self._has_grad(g_dur))
        self.assertTrue(all(g is None for g in g_dose))


if __name__ == '__main__':
    unittest.main()
