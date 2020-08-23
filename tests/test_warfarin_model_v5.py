import unittest

from rl.subjects import Warfarin

class testRLData(unittest.TestCase):
    def test_object_creation(self):
        w = Warfarin()
        print(w)
        self.assertEqual(w.is_terminated, False)
