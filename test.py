from rl.environments import Experiment
from rl.agents import RandomAgent, DQNAgent
from rl.subjects import WarfarinModel

exp = Experiment()

exp.add(agents={'DQN0.2': DQNAgent(filename='WARF_00_00_XX_d_90_dose_9_INR_9_T__dose_change_coef_0.20_DQN_(20,20)_g_0.8_e_func_lr_0.01_buff_900_clr_F_btch_50_vld_0.3'),
                'DQN0.4': DQNAgent(filename='WARF_00_00_XX_d_90_dose_9_INR_9_T__dose_change_coef_0.40_DQN_(20,20)_g_0.8_e_func_lr_0.01_buff_900_clr_F_btch_50_vld_0.3'),
                'DQN0.8': DQNAgent(filename='WARF_00_00_XX_d_90_dose_9_INR_9_T__dose_change_coef_0.80_DQN_(20,20)_g_0.8_e_func_lr_0.01_buff_900_clr_F_btch_50_vld_0.3')},
        subjects={'W1': WarfarinModel(), 'W2': WarfarinModel(), 'W3': WarfarinModel()})

exp.assign([('DQN0.2', 'W1'),
            ('DQN0.4', 'W2'),
            ('DQN0.8', 'W3')])

exp.generate_subjects(number_of_subjects=5)
exp.run()