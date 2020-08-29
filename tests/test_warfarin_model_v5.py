import unittest

from reil.subjects import Warfarin

class testRLData(unittest.TestCase):
    def test_object_creation(self):
        w = Warfarin()
        self.assertEqual(w.is_terminated, False)
