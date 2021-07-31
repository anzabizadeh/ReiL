import unittest
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union
from openpyxl.workbook.workbook import Workbook

import pandas as pd
from openpyxl import load_workbook
from reil.healthcare.dosing_protocols.warfarin import Aurora


class DummyPatients:
    def __init__(self, filename: str) -> None:
        wb = load_workbook(filename, data_only=True)
        self._trajectories = self.read_table(
            wb, 'trajectories', 'trajectories')
        self._trajectories['INR'] = (  # type: ignore
            self._trajectories.INR.apply(  # type: ignore
                round, args=(2,)))
        self._patient_info = self.read_table(
            wb, 'patient_info', 'patient_info', 'ID')

    @staticmethod
    def read_table(
            workbook: Workbook, sheet_name: str, table_name: str,
            index: Optional[str] = None) -> pd.DataFrame:
        ref = workbook[sheet_name]._tables[table_name].ref  # type: ignore
        content: List[List[Any]] = [
            [cell.value for cell in ent]  # type: ignore
            for ent in workbook[sheet_name][ref]  # type: ignore
        ]
        header: List[str] = content[0]
        rest = content[1:]

        df = pd.DataFrame(rest, columns=header)
        if index:
            df = df.set_index(index)  # type: ignore

        return df

    def simulate(
        self, dose: float = -1, interval: int = -1
    ) -> Iterator[Tuple[Dict[Union[str, int], Any], float, int]]:
        for _id, info in self._patient_info.iterrows():
            patient: Dict[Union[str, int], Any] = dict(info.items())
            patient['ID'] = _id
            trajectory = self._trajectories[  # type: ignore
                self._trajectories.ID == _id][  # type: ignore
                ['day', 'INR', 'dose', 'interval']]
            # print(f'patient: {_id}')

            for i in trajectory.index:  # type: ignore
                data = trajectory.loc[:i]  # type: ignore

                patient['day'] = data.day.iat[-1]  # type: ignore
                patient['INR_history'] = list(data.INR)  # type: ignore
                patient['dose_history'] = list(
                    data.dose.iloc[:-1])  # type: ignore
                patient['interval_history'] = list(
                    data.interval.iloc[:-1])  # type: ignore

                yield (
                    patient,
                    data.dose.iat[-1], data.interval.iat[-1])  # type: ignore


class testAurora(unittest.TestCase):
    def test_aurora(self) -> None:
        aurora = Aurora()

        patients = DummyPatients('./tests/data/aurora_sample_dosing.xlsx')

        additional_info = {}
        dose, interval = -1, -1
        _id = -1
        for p, d, i in patients.simulate(
                dose=dose, interval=interval):
            if p['ID'] != _id:
                _id = p['ID']
                additional_info = {}

            dosing_decision, additional_info = aurora.prescribe(
                patient=p, additional_info=additional_info)  # type: ignore
            if additional_info['skip_dose']:
                dosing_decision, additional_info = aurora.prescribe(
                    patient=p, additional_info=additional_info)  # type: ignore
            try:
                self.assertAlmostEqual(d, dosing_decision.dose)
                self.assertEqual(i, dosing_decision.duration)
            except AssertionError:
                print(d, dose, i, interval, '\n', p, '\n', additional_info)
                raise


if __name__ == "__main__":
    unittest.main()
