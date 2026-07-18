"""Tests for the low-rank 2-D joint (dose × duration) head (Paper-3 EA).

Covers: the `_build_actor_logits` hook keeps the default `PPOModel` unchanged;
the pure low-rank cosine math `L[i,j] = s*<P̂_i, Q̂_j>` + ROW-MAJOR flatten (flat
index i*n_dur+j must map to (dose i, dur j), matching the runner's joint action
order); the KL-explosion bound (blowing up P/Q changes logits by <= 2*s);
build/sample/weight round-trip; config round-trip; the action_per_head guard; and
that train_actor traces cleanly. No marginals / gates: the head is a pure coupled
interaction (doc 220 §9.5, 2026-07-13 — u,v + keep-gates removed).
"""
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np
import pytest
import tensorflow as tf

from reil.learners.ppo_learner import (
    LowRankJointHead, PPOLowRankJointModel, PPOLowRankJointNeighborModel,
    PPOModel)

KW = dict(
    input_shape=(5,), actor_learning_rate=1e-4, critic_learning_rate=1e-5,
    actor_layer_sizes=(32, 16), critic_layer_sizes=(32, 16),
    actor_train_iterations=1, critic_train_iterations=1, target_kl=0.1)

D, T, R = 21, 28, 4


def _first(o):
    return o[0] if isinstance(o, (list, tuple)) else o


def test_default_ppomodel_heads_unchanged():
    # the _build_actor_logits hook must not change the default two-head model
    m = PPOModel(action_per_head=(D, T), **KW)
    out = m.actor(tf.zeros((3, 5)))
    assert [o.shape[-1] for o in out] == [D, T]


def test_output_shape_is_flat_joint():
    m = PPOLowRankJointModel(n_dose=D, n_dur=T, rank=R,
                             action_per_head=(D * T,), **KW)
    out = _first(m.actor(tf.zeros((3, 5))))
    assert tuple(out.shape) == (3, D * T)


def test_rowmajor_flatten_matches_grid_math():
    # flat[k] must equal s*<P̂_i, Q̂_j> with k = i*T + j (row-major), i.e. flat
    # index k -> (dose k//T, dur k%T), matching the action ordering
    # `(dose, dur) for dose in doses for dur in durs` built in the runner.
    # Pure cosine head: no marginals, no gates.
    head = LowRankJointHead(D, T, R, name="h")
    h = tf.constant(np.random.RandomState(0).randn(2, 16), dtype=tf.float32)
    flat = head(h).numpy()                                  # (2, D*T) (builds)
    s = head.interaction_scale
    P = head._P(h).numpy().reshape(2, D, R)
    Q = head._Q(h).numpy().reshape(2, T, R)
    P = P / np.linalg.norm(P, axis=-1, keepdims=True)       # unit embeddings
    Q = Q / np.linalg.norm(Q, axis=-1, keepdims=True)
    cos = np.einsum("bik,bjk->bij", P, Q)                   # in [-1, 1]
    expected = (s * cos).reshape(2, D * T)                  # row-major
    assert np.allclose(flat, expected, atol=1e-5)


def test_interaction_bounded_under_large_weights():
    # THE KL-EXPLOSION REGRESSION: blowing up the P/Q embeddings by 1000x must
    # change the logits by at most 2*s (bounded cosine), not ~1000x as the old
    # unbounded dot product did.
    head = LowRankJointHead(D, T, R, name="hb")
    h = tf.constant(np.random.RandomState(1).randn(3, 16), dtype=tf.float32)
    base = head(h).numpy()                                  # builds
    bound = 2.0 * head.interaction_scale
    for lyr in (head._P, head._Q):
        w = lyr.get_weights()
        w[0] = w[0] * 1000.0                                # explode embeddings
        lyr.set_weights(w)
    big = head(h).numpy()
    assert np.max(np.abs(big - base)) <= bound + 1e-3


def test_sample_and_weight_roundtrip():
    m = PPOLowRankJointModel(n_dose=D, n_dur=T, rank=R,
                             action_per_head=(D * T,), **KW)
    s = _first(m.act_sample(tf.zeros((4, 5))))              # graph-mode einsum
    assert tuple(s.shape) == (4,)
    w = m.actor.get_weights()
    m2 = PPOLowRankJointModel(n_dose=D, n_dur=T, rank=R,
                              action_per_head=(D * T,), **KW)
    m2.actor.set_weights(w)
    o1 = _first(m.actor(tf.ones((2, 5))))
    o2 = _first(m2.actor(tf.ones((2, 5))))
    assert np.allclose(o1.numpy(), o2.numpy(), atol=1e-6)


def test_layer_config_roundtrip():
    head = LowRankJointHead(D, T, R, name="h")
    cfg = head.get_config()
    assert cfg["n_dose"] == D and cfg["n_dur"] == T and cfg["rank"] == R
    head2 = LowRankJointHead.from_config(cfg)
    assert (head2.n_dose, head2.n_dur, head2.rank) == (D, T, R)


def test_action_per_head_guard():
    with pytest.raises(ValueError):
        PPOLowRankJointModel(n_dose=D, n_dur=T, action_per_head=(500,), **KW)


def test_train_actor_runs():
    # train_actor must trace + run cleanly on the pure low-rank joint head
    # (inherited PPOModel machinery, single joint head, no custom regulariser).
    m = PPOLowRankJointModel(n_dose=D, n_dur=T, rank=R,
                             action_per_head=(D * T,), **KW)
    rs = np.random.RandomState(2)
    x = tf.constant(rs.randn(8, 5), dtype=tf.float32)
    ai = tf.constant(rs.randint(0, D * T, size=(8, 1)), dtype=tf.int32)
    adv = tf.constant(rs.randn(8), dtype=tf.float32)
    m.train_actor(x, ai, adv)  # must not raise


def test_rank_one_and_two_build_and_run():
    # The pure-joint test uses small rank (1, 2). Both must build + sample.
    for r in (1, 2):
        m = PPOLowRankJointModel(n_dose=D, n_dur=T, rank=r,
                                 action_per_head=(D * T,), **KW)
        out = _first(m.actor(tf.zeros((3, 5))))
        assert tuple(out.shape) == (3, D * T)
        s = _first(m.act_sample(tf.zeros((4, 5))))
        assert tuple(s.shape) == (4,)


def _train_batch(rs):
    x = tf.constant(rs.randn(8, 5), dtype=tf.float32)
    ai = tf.constant(rs.randint(0, D * T, size=(8, 1)), dtype=tf.int32)
    adv = tf.constant(rs.randn(8), dtype=tf.float32)
    return x, ai, adv


def test_neighbor_2d_train_actor_runs_and_updates():
    # The 2-D neighbour model must trace + run train_actor and actually move the
    # actor weights (credit spread over the dose×duration grid neighbourhood).
    m = PPOLowRankJointNeighborModel(
        n_dose=D, n_dur=T, rank=4, action_per_head=(D * T,),
        neighbor_dose_width=1, neighbor_dur_width=1,
        neighbor_dose_decay=0.5, neighbor_dur_decay=0.5, **KW)
    _ = m.actor(tf.zeros((2, 5)))                          # build
    before = [w.numpy().copy() for w in m.actor.trainable_variables]
    rs = np.random.RandomState(5)
    for _ in range(3):
        m.train_actor(*_train_batch(rs))
    after = [w.numpy() for w in m.actor.trainable_variables]
    assert any(not np.allclose(a, b) for a, b in zip(after, before))


def test_neighbor_zero_width_reduces_to_plain_and_config_roundtrip():
    # widths (0,0) => only the (0,0) offset => plain PPO on the joint head.
    m = PPOLowRankJointNeighborModel(
        n_dose=D, n_dur=T, rank=4, action_per_head=(D * T,),
        neighbor_dose_width=0, neighbor_dur_width=0, **KW)
    rs = np.random.RandomState(6)
    m.train_actor(*_train_batch(rs))                       # must not raise
    cfg = m.get_config()
    assert cfg["neighbor_dose_width"] == 0 and cfg["neighbor_dur_width"] == 0
    assert cfg["n_dose"] == D and cfg["rank"] == 4
    # asymmetric widths (dose-only) must also trace + run
    m2 = PPOLowRankJointNeighborModel(
        n_dose=D, n_dur=T, rank=4, action_per_head=(D * T,),
        neighbor_dose_width=2, neighbor_dur_width=0, **KW)
    m2.train_actor(*_train_batch(rs))                      # must not raise
