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


if __name__ == '__main__':
    unittest.main()
