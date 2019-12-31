from rl.subjects.warfarin_model_v5 import WarfarinModel_v5
from rl.rldata import RLData
from copy import deepcopy

class WarfarinLookAhead(WarfarinModel_v5):
    def __init__(self, **kwargs):
        self._lookahead_duration = kwargs.get('lookahead_duration', 7)
        kwargs['max_day'] = kwargs.get('max_day', 90) + self._lookahead_duration
        self._INR_penalty_coef = kwargs.get('INR_penalty_coef', 1)
        self._lookahead_penalty_coef = kwargs.get('lookahead_penalty_coef', 0)
        super().__init__(**kwargs)

    @property
    def is_terminated(self):
        return self._day >= self._max_day - self._lookahead_duration

    def take_effect(self, action, _id=None):
        self._current_dose = action[0]
        self._dosing_intervals.append(self._d_current)
        self._dosing_intervals.popleft()
        self._dose_list.append(self._current_dose)
        self._dose_list.popleft()

        if (self._initial_phase_duration == -1 and self._therapeutic_range[0] <= self._INR[-1] <= self._therapeutic_range[1]) \
            or self._day == self._initial_phase_duration:
                self._phase = 'maintenance'
        if self._phase == 'initial':
            self._d_current = 1  # action.value[1]
            self._patient.dose = {self._day: self._current_dose}
        else:
            self._d_current = min(self._maintenance_day_interval, self._max_day-self._day)
            self._patient.dose = dict(tuple((i + self._day, self._current_dose) for i in range(self._d_current)))

        d = self._day
        self._day += self._d_current

        self._INR.append(self._patient.INR(self._day)[-1])
        self._INR.popleft()

        try:
            if self.exchange_protocol['take_effect'] == 'standard':
                temp_patient = deepcopy(self._patient)
                temp_patient.dose = dict(tuple((i + d, self._current_dose) for i in range(1, self._lookahead_duration + 1)))
                lookahead_penalty = -sum(((2 / self._INR_range * (self._INR_mid - INRi)) ** 2
                                        for INRi in temp_patient.INR(list(range(d + 1, d + self._lookahead_duration + 1)))))

                INR_penalty = -sum(((2 / self._INR_range * (self._INR_mid - self._INR[-2] - (self._INR[-1]-self._INR[-2])/self._d_current*j)) ** 2
                                    for j in range(1, self._d_current + 1)))  # negative squared distance as reward (used *2/range to normalize)
                dose_change_penalty = - self._dose_change_penalty_func(self._dose_list)
                reward = self._INR_penalty_coef * INR_penalty \
                        + self._dose_change_penalty_coef * dose_change_penalty \
                        + self._lookahead_penalty_coef * lookahead_penalty
            elif self.exchange_protocol['take_effect'] == 'no_reward':
                reward = 0
        except TypeError:
            reward = 0

        return RLData(reward, normalizer=lambda x: x)

if __name__ == '__main__':
    w = WarfarinLookAhead(lookahead_penalty_coef=1, INR_penalty_coef=0)
    for i in range(90):
        reward = w.take_effect(RLData(10, lower=0, upper=15))
        print(reward[0], w._INR[-1])