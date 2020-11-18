from typing import Any, Dict

from reil.utils import Feature


class HealthMathModel:
    def setup(self, **arguments):
        raise NotImplementedError

    def run(self, **inputs) -> Dict[str, Any]:
        raise NotImplementedError
