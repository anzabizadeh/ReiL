import unittest

from rl.rldata import RLData
from random import randint, sample

class testRLData(unittest.TestCase):
    def test_numerical_list_creation(self):
        data = [randint(-100, 100) for _ in range(10)]
        rl_value = RLData(data)
        self.assertEqual(rl_value._value, data)
        self.assertEqual(rl_value.value, data)

    def test_numerical_list_indexing(self):
        data = [randint(-100, 100) for _ in range(10)]
        rl_value = RLData(data)
        for i in range(len(data)):
            self.assertEqual(rl_value[i], data[i])

        index = randint(0, len(data))
        self.assertEqual(rl_value[index],
                         data[index],
                         f'index for {index}')

        slice_from, slice_to = sorted([randint(0, len(data)),
                               randint(0, len(data))])
        self.assertEqual(rl_value[slice_from:slice_to].value,
                         data[slice_from:slice_to],
                         f'slicing for {slice_from}:{slice_to}')

    def test_numerical_list_modification(self):
        data_numerical = [randint(-100, 100) for _ in range(10)]
        rl_value_numerical = RLData(data_numerical, lower=-100, upper=100)

        # slice of a list - numerical
        # Normal
        new_value = [randint(-100, 100) for _ in range(2)]
        result = rl_value_numerical[:2].value + new_value + rl_value_numerical[4:].value
        rl_value_numerical[2:4] = new_value
        self.assertEqual(rl_value_numerical.value, result)

        # Exception due to range
        new_value = [randint(-150, -100) for _ in range(2)]
        result = rl_value_numerical[:2].value + new_value + rl_value_numerical[4:].value
        with self.assertRaises(ValueError):
            rl_value_numerical[2:4] = new_value

        # one index of a list
        data_numerical = [randint(-100, 100) for _ in range(10)]
        rl_value_numerical = RLData(data_numerical, lower=-100, upper=100)
        new_value = randint(-100, 100)
        index = randint(0, len(data_numerical))
        data_numerical[index] = new_value
        rl_value_numerical[index] = new_value
        self.assertEqual(rl_value_numerical.value, data_numerical)

        with self.assertRaises(ValueError):
            rl_value_numerical[0] = 1000

        with self.assertRaises(TypeError):
            rl_value_numerical['x'] = 10

    def test_categorical_list_modification(self):
        # slice of a list - categorical
        # Normal
        data_categorical = sample(list('ABCDEFGHIJKLMNOPQRSTUVWXYZ'), 10)
        rl_value_categorical = RLData(data_categorical)
        new_value = sample(rl_value_categorical.categories, 2)
        result = rl_value_categorical[:2].value + new_value + rl_value_categorical[4:].value
        rl_value_categorical[2:4] = new_value
        self.assertEqual(rl_value_categorical.value, result)

        # Exception due to category
        new_value = sample(list('abcd'), 2)
        result = rl_value_categorical[:2].value + new_value + rl_value_categorical[4:].value
        with self.assertRaises(ValueError):
            rl_value_categorical[2:4] = new_value

        with self.assertRaises(ValueError):
            rl_value_categorical[0] = 'a'

        with self.assertRaises(TypeError):
            rl_value_categorical['x'] = rl_value_categorical[-1]

    def test_numerical_list_normalizer(self):
        data = [randint(-100, 100) for _ in range(10)]
        rl_value = RLData(data)
        lower = min(data)
        upper = max(data)
        values_range = upper - lower
        expected_output = [(i - lower) / values_range for i in data]
        self.assertEqual(rl_value.normalize(), RLData(expected_output))

        # slice of a list - numerical
        new_value = [randint(lower, upper) for _ in range(2)]
        result = rl_value[:2].value + new_value + rl_value[4:].value
        expected_output = [(i - lower) / values_range for i in result]
        rl_value[2:4] = new_value
        self.assertEqual(rl_value.normalize().value, expected_output)

    def test_categorical_list_normalizer(self):
        raise NotImplementedError

    def test_categorical_dict_modification(self):
        # slice of a list - categorical
        # Normal
        data_categorical = dict(zip(sample(list('ABCDEFGHIJKLMNOPQRSTUVWXYZ'), 10), (randint(-100, 100) for _ in range(10))))
        rl_value_categorical = RLData(data_categorical)
        index = sample(rl_value_categorical.keys(), 1)[0]
        
        rl_value_categorical._lower[index] = -100
        rl_value_categorical._upper[index] = 100
        rl_value_categorical[index] = 12.5
        self.assertEqual(rl_value_categorical[index].value, [12.5])





if __name__ == "__main__":
    unittest.main()
