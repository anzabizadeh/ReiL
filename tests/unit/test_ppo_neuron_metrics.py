import unittest

import numpy as np
import tensorflow as tf

from reil.learners.ppo_learner import PPOModel
from reil.utils.tf_utils import entropy


class TestPPONeuronMetrics(unittest.TestCase):
    """Covers the action-forging diagnostics added on
    feat/action-forging-instrumentation: per-neuron L2 vector, the
    `n_actions_alive` derived count, and the batch policy entropy."""

    def _build_model(self, n_actions: int = 5) -> PPOModel:
        return PPOModel(
            input_shape=(4,),
            action_per_head=(n_actions,),
            actor_learning_rate=1e-4,
            critic_learning_rate=1e-5,
            actor_layer_sizes=(8,),
            critic_layer_sizes=(8,),
            actor_train_iterations=1,
            critic_train_iterations=1,
            target_kl=0.02,
            clip_ratio=0.2,
            regularizer_coef=0.0,
        )

    def test_per_neuron_l2_vector_shape_and_values(self):
        model = self._build_model(n_actions=5)
        vec = model.per_neuron_l2_vector().numpy()
        self.assertEqual(vec.shape, (5,))
        self.assertTrue(np.all(np.isfinite(vec)))
        # Output-layer init = keras.initializers.Constant(0.1) (see PPOModel
        # logit_heads); norms must be strictly positive at init.
        self.assertTrue(np.all(vec > 0.0))

    def test_n_actions_alive_at_init(self):
        model = self._build_model(n_actions=5)
        vec = model.per_neuron_l2_vector().numpy()
        # All neurons start with constant-0.1 weights → all alive.
        self.assertEqual(int((vec > 1e-4).sum()), 5)

    def test_batch_policy_entropy_finite(self):
        model = self._build_model(n_actions=5)
        x = tf.constant(np.random.RandomState(0).randn(4, 4).astype(np.float32))
        logits = model.actor(x, training=False)
        if isinstance(logits, (list, tuple)):
            logits = tf.concat(logits, axis=1)
        e = float(tf.reduce_mean(entropy(logits)))
        self.assertTrue(np.isfinite(e))
        # Uniform over 5 actions → entropy ≈ ln(5) ≈ 1.609. With constant
        # weights init the policy starts near uniform, so e should be close.
        self.assertGreater(e, 1.5)
        self.assertLess(e, np.log(5) + 1e-6)


if __name__ == '__main__':
    unittest.main()
