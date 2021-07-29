import unittest

from reil.stateful import Stateful
from reil.utils.instance_generator_batch import InstanceGeneratorBatch


class testInstanceGeneratorBatch(unittest.TestCase):
    def test_no_save(self):
        obj = Stateful(name='test')
        ig = InstanceGeneratorBatch(
            obj=obj,
            instance_counter_stops=(3, 5, 10, 20),
            save_instances=True,
            use_existing_instances=True
        )

        for _ in range(10):
            for i, a in ig:
                print(i, a._name)
            print('----done----')


if __name__ == "__main__":
    unittest.main()
