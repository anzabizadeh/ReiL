from rl.environments import Experiment
from rl.agents import DQNAgent, WarfarinAgent
from rl.subjects import WarfarinModel, WarfarinModel_v4

# age=70,
# CYP2C9='*1/*3',
# VKORC1='A/A',
max_day = 90
patient_selection = 'ravvaz'
dose_history = 9
INR_history = 9
dose_change_penalty_coef = 0.2
dose_change_penalty_func = lambda x: 0.2*int(x[-2]!=x[-1])
randomized = True
extended_state = True

agents={
        'aurora': WarfarinAgent()
        # '0.20_2_day': DQNAgent(path='./WARFV4_00_00_XX_d_90_dose_9_INR_9_T__dose_change_coef_0.20_DQN_(20,20)_g_0.95_e_func_lr_0.01_buff_900_clr_F_btch_50_vld_0.3.data', filename='protocol'),
        # '0.40_2_day': DQNAgent(path='./WARFV4_00_00_XX_d_90_dose_9_INR_9_T__dose_change_coef_0.40_DQN_(20,20)_g_0.95_e_func_lr_0.01_buff_900_clr_F_btch_50_vld_0.3.data', filename='protocol'),
        # '0.50_4_day': DQNAgent(path='./WARFV4_00_00_XX_d_90_dose_9_INR_9_T__dose_change_coef_0.50_DQN_(20,20)_g_0.95_e_func_lr_0.01_buff_900_clr_F_btch_50_vld_0.3.data', filename='protocol'),
        # '0.60_2_day': DQNAgent(path='./WARFV4_00_00_XX_d_90_dose_9_INR_9_T__dose_change_coef_0.60_DQN_(20,20)_g_0.95_e_func_lr_0.01_buff_900_clr_F_btch_50_vld_0.3.data', filename='protocol'),
        # '0.80_2_day': DQNAgent(path='./WARFV4_00_00_XX_d_90_dose_9_INR_9_T__dose_change_coef_0.80_DQN_(20,20)_g_0.95_e_func_lr_0.01_buff_900_clr_F_btch_50_vld_0.3.data', filename='protocol'),
        # '1.00_2_day': DQNAgent(path='./WARFV4_00_00_XX_d_90_dose_9_INR_9_T__dose_change_coef_1.00_DQN_(20,20)_g_0.95_e_func_lr_0.01_buff_900_clr_F_btch_50_vld_0.3.data', filename='protocol')
        }

subjects={'Warfarin_v4_extended_state': WarfarinModel_v4(max_day=max_day,
                                patient_selection=patient_selection,
                                dose_history=dose_history,
                                INR_history=INR_history,
                                randomized=randomized,
                                dose_change_penalty_coef=dose_change_penalty_coef,
                                dose_change_penalty_func=dose_change_penalty_func,
                                extended_state=extended_state,
                                save_patient=False)}

exp = Experiment()

exp.add(agents=agents, subjects=subjects)

exp.generate_subjects(number_of_subjects=10)

for agent in agents:
        exp.assign([(agent, 'Warfarin_v4_extended_state')])
        print(agent)
        exp.run()
        exp.divest([(agent, 'Warfarin_v4_extended_state')])
