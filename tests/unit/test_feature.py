import unittest
from random import randint, sample

from reil.datatypes import feature


class testFeature(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._numerical_data = [randint(-100, 100) for _ in range(10)]
        cls._categorical_data = sample(list('ABCDEFGHIJKLMNOPQRSTUVWXYZ'), 10)

    def test_create_feature(self):
        rl_value = feature.Feature(
            name='test', value=self._numerical_data, is_numerical=True)
        self.assertEqual(rl_value.name, 'test')
        self.assertEqual(rl_value.value, self._numerical_data)
        self.assertEqual(rl_value.is_numerical, True)

        lower = min(self._numerical_data)
        upper = max(self._numerical_data)

        def midpoint(f: feature.Feature) -> float:
            return (f.lower + f.upper)/2  # type: ignore

        numerical = feature.FeatureGenerator.continuous(
            name='num', lower=lower,
            upper=upper, generator=midpoint, randomized=True)  # type: ignore

        self.assertEqual(numerical.name, 'num')
        self.assertEqual(numerical().value, (lower + upper)/2)
        self.assertEqual(numerical.is_numerical, True)

        categorical = feature.FeatureGenerator.categorical(
            name='cat', categories=('A', 'B', 'C'), probabilities=(.3, .6, .1),
            generator=lambda f: 'A', randomized=True)

        self.assertEqual(categorical.name, 'cat')
        self.assertEqual(categorical().value, 'A')
        self.assertEqual(categorical.is_numerical, False)

    def test_feature_config(self):
        rl_value = feature.Feature(
            name='test', value=self._numerical_data, is_numerical=True)
        self.assertEqual(
            feature.Feature.from_config(rl_value.get_config()), rl_value)

        lower = min(self._numerical_data)
        upper = max(self._numerical_data)

        def midpoint(f: feature.Feature) -> float:
            return (f.lower + f.upper)/2  # type: ignore

        numerical = feature.FeatureGenerator.continuous(
            name='num', lower=lower,
            upper=upper, generator=midpoint, randomized=True)  # type: ignore

        self.assertEqual(
            feature.FeatureGenerator.from_config(
                numerical.get_config()), numerical)

        categorical = feature.FeatureGenerator.categorical(
            name='cat', categories=('A', 'B', 'C'), probabilities=(.3, .6, .1),
            generator=lambda f: 'A', randomized=True)

        self.assertEqual(
            feature.FeatureGenerator.from_config(
                categorical.get_config()), categorical)

        fg_set = feature.FeatureGeneratorSet((numerical, categorical))

        self.assertEqual(
            feature.FeatureGeneratorSet.from_config(
                fg_set.get_config()), fg_set)

    def test_add_features(self):
        rl_values = [feature.Feature(
            name='test', value=v, is_numerical=True)
            for v in self._numerical_data]
        rl_new = sum(rl_values[1:], rl_values[0])
        self.assertEqual(rl_new.name, 'test')
        self.assertEqual(rl_new.value, sum(self._numerical_data))
        self.assertEqual(rl_new.is_numerical, True)

    def test_add_numerical(self):
        numericals = feature.FeatureGenerator.continuous(name='test')
        rl_values = [numericals(v) for v in self._numerical_data]

        rl_new = sum(rl_values[1:], rl_values[0])
        self.assertEqual(rl_new.name, 'test')
        self.assertEqual(rl_new.value, sum(self._numerical_data))
        self.assertEqual(rl_new.is_numerical, True)

        rl_values = [numericals((v,)) for v in self._numerical_data]

        rl_new = sum(rl_values[1:], rl_values[0])
        self.assertEqual(rl_new.name, 'test')
        self.assertEqual(rl_new.value, tuple(self._numerical_data))
        self.assertEqual(rl_new.is_numerical, True)

    def test_add_categorical(self):
        categoricals = feature.FeatureGenerator.categorical(
            name='test', categories=tuple(self._categorical_data))
        rl_values = [categoricals(v) for v in self._categorical_data]
        self.assertEqual(
            rl_values[0].normalized,
            tuple([1] + [0]*(len(self._categorical_data) - 2)))
        self.assertEqual(
            rl_values[-1].normalized,
            tuple([0]*(len(self._categorical_data) - 1)))

        rl_new = sum(rl_values[1:], rl_values[0])
        self.assertEqual(rl_new.name, 'test')
        self.assertEqual(rl_new.value, ''.join(self._categorical_data))
        self.assertEqual(rl_new.is_numerical, False)

        rl_values = [categoricals((v,)) for v in self._categorical_data]

        rl_new = sum(rl_values[1:], rl_values[0])
        self.assertEqual(rl_new.name, 'test')
        self.assertEqual(rl_new.value, tuple(self._categorical_data))
        self.assertEqual(rl_new.is_numerical, False)

    def test_allow_missing(self):
        categoricals = feature.FeatureGenerator.categorical(
            name='test', categories=tuple(self._categorical_data),
            allow_missing=True)
        rl_values = [categoricals(v) for v in self._categorical_data]
        self.assertEqual(
            rl_values[0].normalized,
            tuple([1] + [0]*(len(self._categorical_data) - 1)))
        self.assertEqual(
            rl_values[-1].normalized,
            tuple([0]*(len(self._categorical_data) - 1) + [1]))
        self.assertEqual(
            categoricals(feature.MISSING).normalized,
            tuple([0]*(len(self._categorical_data))))

    def test_feature_set(self):
        categoricals = feature.FeatureGenerator.categorical(
            name='cats', categories=tuple(self._categorical_data))
        numericals = feature.FeatureGenerator.continuous(
            name='nums', lower=min(self._numerical_data),
            upper=max(self._numerical_data))

        with self.assertRaises(KeyError):
            feature.FeatureSet(
                [categoricals(v) for v in self._categorical_data])
            # print(rl_set.categories)

        reilset = feature.FeatureSet([
            categoricals(self._categorical_data[0]),
            numericals(self._numerical_data[0]),
        ])

        self.assertEqual(reilset.lower,
                         {reilset['cats'].name: reilset['cats'].lower,
                          reilset['nums'].name: reilset['nums'].lower})
        self.assertEqual(reilset.upper,
                         {reilset['cats'].name: reilset['cats'].upper,
                          reilset['nums'].name: reilset['nums'].upper})
        self.assertEqual(
            reilset.categories,
            {reilset['cats'].name: reilset['cats'].categories,
             reilset['nums'].name: reilset['nums'].categories})

        self.assertEqual(reilset.normalized.flattened,
                         list(reilset['cats'].normalized) +  # type: ignore
                         [reilset['nums'].normalized])

        self.assertEqual(
            reilset,
            feature.FeatureSet([categoricals(self._categorical_data[0])]) +
            feature.FeatureSet([numericals(self._numerical_data[0])])
            )

    def test_change_to_missing(self):
        categoricals = feature.FeatureGenerator.categorical(
            name='test', categories=tuple(self._categorical_data),
            allow_missing=True)
        self.assertEqual(
            feature.change_to_missing(categoricals(  # type: ignore
                self._categorical_data[0])).normalized,
            tuple([0]*len(self._categorical_data)))

        set_ = feature.FeatureSet(
            feature.FeatureGenerator.categorical(
                name=v, categories=tuple(self._categorical_data),
                allow_missing=True)(v)
            for v in self._categorical_data)
        missing_set = feature.change_set_to_missing(set_)
        self.assertEqual(
            next(iter(missing_set)).normalized,
            tuple([0]*len(self._categorical_data)))

        with self.assertRaises(TypeError):
            feature.change_to_missing(
                feature.FeatureGenerator.categorical(  # type: ignore
                    name='test', categories=tuple(self._categorical_data),
                    allow_missing=False)(self._categorical_data[0]))

        with self.assertRaises(ValueError):
            feature.change_to_missing(
                feature.FeatureGenerator.categorical(  # type: ignore
                    name='test',
                    allow_missing=True)(self._categorical_data[0]))


if __name__ == "__main__":
    unittest.main()
