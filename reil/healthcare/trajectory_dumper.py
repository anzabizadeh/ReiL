from __future__ import annotations

import pathlib
import re
import pandas as pd
from reil.datatypes.feature import FeatureArray
from reil.datatypes.feature_array_dumper import FeatureArrayDumper


class TrajectoryDumper(FeatureArrayDumper):
    @staticmethod
    def _dump(
            component: FeatureArray,
            filename: str, path: pathlib.PurePath) -> bool:
        '''Write stats to file.'''

        component_dict = component.value
        measure_names = re.findall(
            'daily_((?!dose).+?)_history', ' '.join(component_dict))

        temp = pd.DataFrame({m: component_dict[f'daily_{m}_history']
                             for m in measure_names})
        temp.drop(temp.tail(1).index, inplace=True)  # type: ignore
        temp['dose'] = component_dict['daily_dose_history']
        temp['decision_points'] = [
            a for t in component_dict['interval_history']
            for a in ([1] + [0] * (t-1))]

        temp['day'] = temp.index + 1  # type: ignore

        for k, v in component_dict.items():
            if k not in ['daily_dose_history',
                         'interval_history',
                         'day'] + [f'daily_{m}_history'
                                   for m in measure_names]:
                temp[k] = v

        try:
            fname = pathlib.Path(path / filename)
            header = not fname.exists()
            with open(fname, 'a+', newline='') as f:
                temp.to_csv(f, mode='a+', header=header)
        except PermissionError:
            return False

        return True
