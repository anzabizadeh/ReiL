import unittest

from reil.subjects.healthcare import Warfarin


class testWarfarin(unittest.TestCase):
    def test_object_creation(self):
        w = Warfarin()
        self.assertEqual(w.is_terminated, False)

    def test_get_next_interval(self):
        interval = [5]
        w = Warfarin(interval=interval)
        for _ in range(5):
            self.assertEqual(w._current_interval, 5)
            w._determine_interval_info()

        interval = [-5, 2]
        w = Warfarin(interval=interval)
        for _ in range(5):
            self.assertEqual(w._current_interval, 5)
            w._determine_interval_info()

        w._decision_points_index = 1
        w._decision_points_INR_history[w._decision_points_index] = 2.5
        w._determine_interval_info()
        for _ in range(5):
            self.assertEqual(w._current_interval, 2)
            w._determine_interval_info()

        interval = [1, 2, 3, 4, 5]
        w = Warfarin(interval=interval)
        for i in interval:
            self.assertEqual(w._current_interval, i)
            w._determine_interval_info()
