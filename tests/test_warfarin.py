import unittest

from reil.subjects import Warfarin

class testWarfarin(unittest.TestCase):
    def test_object_creation(self):
        w = Warfarin()
        self.assertEqual(w.is_terminated, False)

    def test_get_next_interval(self):
        interval = [5]
        w = Warfarin(interval=interval)
        for _ in range(5):
            self.assertEqual(w._get_next_interval(), 5)

        interval = [-5, 2]
        w = Warfarin(interval=interval)
        for _ in range(5):
            self.assertEqual(w._get_next_interval(), 5)

        w._decision_points_index = 1
        w._decision_points_INR_history[w._decision_points_index] = 2.5
        for _ in range(5):
            self.assertEqual(w._get_next_interval(), 2)

        interval = [1, 2, 3, 4, 5]
        w = Warfarin(interval=interval)
        for i in interval:
            self.assertEqual(w._get_next_interval(), i)