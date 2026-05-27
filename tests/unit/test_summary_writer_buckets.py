import pickle
import unittest

import tensorflow as tf

from reil.utils.tf_utils import SummaryWriter


class TestSummaryWriterBuckets(unittest.TestCase):
    """Covers SummaryWriter.set_buckets() and its pickle round-trip on
    feat/action-forging-instrumentation."""

    def test_set_buckets_updates_map(self):
        w = SummaryWriter()
        w.set_data_types({'dose': 'histogram', 'kl': 'scalar'})
        w.set_buckets({'dose': 21})
        self.assertEqual(w._buckets, {'dose': 21})

    def test_pickle_roundtrip_preserves_buckets(self):
        w = SummaryWriter()
        w.set_data_types({'dose': 'histogram'})
        w.set_buckets({'dose': 21})
        restored: SummaryWriter = pickle.loads(pickle.dumps(w))
        self.assertEqual(restored._buckets, {'dose': 21})

    def test_legacy_pickle_without_buckets_loads(self):
        # Simulate a pickle written by code that pre-dates `buckets`.
        w = SummaryWriter()
        w.set_data_types({'dose': 'histogram'})
        state = w.__getstate__()
        state.pop('buckets')  # drop the new field
        restored = SummaryWriter()
        restored.__setstate__(state)
        self.assertEqual(restored._buckets, {})

    def test_write_with_bucketed_histogram(self):
        # Smoke test: writing a histogram with a custom bucket count must
        # not raise. We use a noop writer (no tensorboard_path) so this
        # is a pure code-path check.
        w = SummaryWriter()
        w.set_data_types({'dose': 'histogram'})
        w.set_buckets({'dose': 5})
        w.write({'dose': tf.constant([0, 1, 2, 3, 4], dtype=tf.int32)},
                iteration=0)


if __name__ == '__main__':
    unittest.main()
