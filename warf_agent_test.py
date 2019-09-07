from rl.subjects import WarfarinModel_v4
from rl.agents import WarfarinAgent

w = WarfarinModel_v4(characteristics={'age': 61.3, 'CYP2C9': '*1/*2',
                     'VKORC1G': 'G/G'},
                     list_of_characteristics={'age': (20, 100)}, patient_selection='', extended_state=True)
a = WarfarinAgent()
INRs = [
    1, 1.5, 2, 2.5, 2.7, 2.8, 3.3, 3.4, 3.8, 3.8, 4, 4, 3.9, 4, 4, 4.3, 3.6, 3, 3.6, 3.5, 3.7, 3.9, 4, 3.3, 3, 3.3, 3.4, 3.4, 3.5, 3.6, 3.5, 3.5, 3.5, 3.7, 3.5, 3.6, 3.7, 3.5, 3.6, 3.4, 3.4, 3.3, 3.4, 3.5, 3.3, 3.4, 3.1, 3.3, 3.4, 3.4, 3.3, 3.3, 3.2, 3.1, 3.1, 3.4, 3.1, 3.2, 3.1, 2.8, 3, 3, 2.9, 2.9, 2.9, 3, 2.8, 3, 2.9, 3, 2.8, 2.9, 2.9, 2.9, 2.8, 2.8, 2.9, 2.8, 2.9, 2.9, 2.9, 2.8, 2.8, 2.9, 2.9, 2.9, 2.7, 2.8, 2.8, 2.8
]

for i in INRs:
    action = a.act(w.state)
    w._INR.append(i)
    w._INR.popleft()
    w._day += 1
    print(action.value[0], w.state.value.INRs[-1])
