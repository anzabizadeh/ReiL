"""Unit tests for reward `ReilFunction`s used by the warfarin experiments.

Covers (a) the tuple-input regression that broke the EXP-C2-RW1 LINEAR cells,
and (b) the EXP-C2-RW1 reward-shape variants reducing to the canonical reward
at their degenerate settings.
"""
import unittest

from reil.utils import reil_functions as rf

COMMON = dict(y_var_name='x', length=-1, multiplier=-1.0, interpolate=False,
              center=2.5, band_width=1.0, exclude_first=False)


class TestRewardTupleInput(unittest.TestCase):
    """Regression: distance rewards must accept a tuple `y`, not just a list.

    `NormalizedDistance` / `PercentInRange` used ``_y = [0.0] + y``, which
    raises ``TypeError: can only concatenate list (not "tuple") to list`` when
    the subject passes a tuple. Every EXP-C2-RW1 LINEAR cell crashed on this
    until it was changed to ``[0.0, *y]`` (matching `NormalizedSquareDistance`).
    """

    def test_normalized_distance_accepts_tuple(self):
        fn = rf.NormalizedDistance(name='nd', amplifying_factor=1.05, **COMMON)
        y = [2.0, 2.5, 3.0, 1.8]
        self.assertAlmostEqual(fn._default_function(list(y)),
                               fn._default_function(tuple(y)), places=12)

    def test_percent_in_range_accepts_tuple(self):
        fn = rf.PercentInRange(name='pir', y_var_name='x', length=-1,
                               multiplier=1.0, interpolate=False,
                               acceptable_range=(2, 3), exclude_first=False)
        y = [2.0, 2.5, 3.0, 1.8]
        self.assertAlmostEqual(fn._default_function(list(y)),
                               fn._default_function(tuple(y)), places=12)


class TestRewardShapeVariants(unittest.TestCase):
    """EXP-C2-RW1 reward classes reduce to the canonical squared reward at
    their degenerate settings, so each reward-shape contrast is clean."""

    def setUp(self):
        self.y = [1.5, 2.0, 2.5, 3.0, 3.5, 2.2, 2.8]
        self.base = rf.NormalizedSquareDistance(
            name='b', amplifying_factor=1.05, **COMMON)

    def test_deadband_zero_tolerance_equals_square(self):
        db = rf.DeadbandSquareDistance(
            name='d', amplifying_factor=1.05, tolerance=0.0, **COMMON)
        self.assertAlmostEqual(db._default_function(self.y),
                               self.base._default_function(self.y), places=12)

    def test_deadband_inside_band_is_zero(self):
        db = rf.DeadbandSquareDistance(
            name='d', amplifying_factor=1.05, tolerance=0.5, **COMMON)
        # all values within [2.0, 3.0] = [center +/- tolerance] -> no penalty
        self.assertEqual(db._default_function([2.0, 2.2, 2.5, 2.8, 3.0]), 0.0)

    def test_asymmetric_symmetric_equals_square(self):
        a = rf.AsymmetricSquareDistance(
            name='a', amplifying_factor=1.05,
            under_weight=1.0, over_weight=1.0, **COMMON)
        self.assertAlmostEqual(a._default_function(self.y),
                               self.base._default_function(self.y), places=12)

    def test_asymmetric_overweights_high_side_only(self):
        a = rf.AsymmetricSquareDistance(
            name='a', amplifying_factor=1.05,
            under_weight=1.0, over_weight=4.0, **COMMON)
        # below center: identical to base; above center: 4x base.
        self.assertAlmostEqual(a._default_function([1.5]),
                               self.base._default_function([1.5]), places=12)
        self.assertAlmostEqual(a._default_function([3.5]),
                               4 * self.base._default_function([3.5]), places=12)

    def test_severe_excursion_zero_weight_equals_square(self):
        s = rf.SevereExcursionSquareDistance(
            name='s', amplifying_factor=1.05, hi=4.0, hi_weight=0.0, **COMMON)
        self.assertAlmostEqual(s._default_function(self.y),
                               self.base._default_function(self.y), places=12)

    def test_severe_excursion_inert_below_threshold(self):
        s = rf.SevereExcursionSquareDistance(
            name='s', amplifying_factor=1.05, hi=4.0, hi_weight=4.0, **COMMON)
        # every value at or below hi is priced exactly as the anchor prices it,
        # which is what distinguishes this from AsymmetricSquareDistance.
        for v in (1.5, 2.5, 3.0, 3.9, 4.0):
            self.assertAlmostEqual(s._default_function([v]),
                                   self.base._default_function([v]), places=12)

    def test_severe_excursion_adds_hinge_above_threshold(self):
        s = rf.SevereExcursionSquareDistance(
            name='s', amplifying_factor=1.05, hi=4.0, hi_weight=4.0, **COMMON)
        # at INR 5: base (2.5-5)^2 plus 4*(5-4)^2, under the same normalization.
        extra = s._default_function([5.0]) - self.base._default_function([5.0])
        scale = (2.0 / COMMON['band_width']) ** 2
        self.assertAlmostEqual(extra, 4.0 * (5.0 - 4.0) ** 2 * scale, places=12)

    def test_severe_excursion_is_registered_for_warfarin(self):
        from reil.healthcare.subjects.warfarin import reward_definitions
        for name in ('hipen_hi4_w4', 'hipen_hi3p5_w4'):
            self.assertIn(name, reward_definitions)
            fn, state = reward_definitions[name]
            self.assertIsInstance(fn, rf.SevereExcursionSquareDistance)
            self.assertEqual(state, 'recent_daily_INR')
            # IJ9 rides on the eta=1.0 anchor, not the eta=1.05 RW1 variants.
            self.assertEqual(fn.amplifying_factor, 1.0)


if __name__ == '__main__':
    unittest.main()
