import unittest
from typing import Any, Dict, Iterator, Union

import pandas as pd
from openpyxl import load_workbook
from reil.healthcare.dosing_protocols.warfarin import Aurora


class DummyPatients:
    def __init__(self, filename: str) -> None:
        wb = load_workbook(filename, data_only=True)
        self._trajectories = self.read_table(
            wb, 'trajectories', 'trajectories')
        self._trajectories['INR'] = self._trajectories.INR.apply(
            round, args=(2,))
        self._patient_info = self.read_table(
            wb, 'patient_info', 'patient_info', 'ID')

    @staticmethod
    def read_table(workbook, sheet_name, table_name, index=None):
        ref = workbook[sheet_name]._tables[table_name].ref
        content = [
            [cell.value for cell in ent]
            for ent in workbook[sheet_name][ref]
        ]
        header = content[0]
        rest = content[1:]

        df = pd.DataFrame(rest, columns=header)
        if index:
            df = df.set_index(index)

        return df

    def simulate(self, dose: float = -1, interval: int = -1) -> Iterator:
        for _id, info in self._patient_info.iterrows():
            patient: Dict[Union[str, int], Any] = dict(info.items())
            patient['ID'] = _id
            trajectory = self._trajectories[self._trajectories.ID == _id][
                ['day', 'INR', 'dose', 'interval']]
            # print(f'patient: {_id}')

            for i in trajectory.index:
                data = trajectory.loc[:i]  # type: ignore

                patient['day'] = data.day.iat[-1]
                patient['INR_history'] = list(data.INR)
                patient['dose_history'] = list(data.dose.iloc[:-1])
                patient['interval_history'] = list(data.interval.iloc[:-1])

                yield patient, data.dose.iat[-1], data.interval.iat[-1]


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
# Note: "stable dose" for days 3 and 4...
            dose, interval, additional_info = aurora.prescribe(
                patient=p, additional_info=additional_info)
            if additional_info['skip_dose']:
                dose, interval, additional_info = aurora.prescribe(
                    patient=p, additional_info=additional_info)
            try:
                self.assertAlmostEqual(d, dose)
                self.assertEqual(i, interval)
            except AssertionError:
                print(d, dose, i, interval, '\n', p, '\n', additional_info)
                raise


if __name__ == "__main__":
    unittest.main()
