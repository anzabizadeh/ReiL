import math
import random
from typing import Any

import numpy as np
from reil.utils import Feature
from scipy import stats


def random_truncated_normal(f: Feature) -> float:
    return min(max(np.random.normal(f.mean, f.stdev), f.lower), f.upper)  # type:ignore


def random_uniform(f: Feature):
    return np.random.uniform(f.lower, f.upper)  # type:ignore


def random_categorical(f: Feature):
    if f.probabilities is None:
        return random.choice(f.categories)  # type:ignore
    else:
        return np.random.choice(f.categories, 1, p=f.probabilities)[0]  # type:ignore


def random_truncated_lnorm(f: Feature) -> float:
    # capture 50% of the data.  This restricts the log values to a "reasonable" range
    quartileRange = (0.25, 0.75)
    lnorm = stats.lognorm(f.stdev, scale=math.exp(f.mean))  # type:ignore
    qValues = lnorm.ppf(quartileRange)
    values = list(v for v in lnorm.rvs(size=1000)
                  if (v > qValues[0]) & (v < qValues[1]))
    return random.sample(values, 1)[0]


def get_argument(x: Any, y: Any) -> Any:
    return x if x is not None else y
