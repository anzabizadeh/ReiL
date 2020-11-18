from typing import Any, Dict

from reil.subjects.healthcare.mathematical_models import HealthMathModel
from reil.utils import Feature


class Patient:
    feature_set: Dict[str, Feature] = {}

    def __init__(self, model: HealthMathModel, **feature_values: Any) -> None:
        for k, v in self.feature_set.items():
            if k in feature_values:
                self.feature_set[k] = v
            else:
                self.feature_set[k].generate()

        self._model = model
        self._model.setup(**self.feature_set)

    def generate(self) -> None:
        for v in self.feature_set.values():
            v.generate()

        self._model.setup(**self.feature_set)

    def model(self, **inputs) -> Dict[str, Any]:
        return self._model.run(**inputs)
