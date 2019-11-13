# Disable GPU before loading tensorflow
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

from rl.subjects import WarfarinModel_v5
from rl.agents import DQNAgent, WarfarinClusterAgent, WarfarinAgent
from rl.environments import Experiment
import tensorflow as tf
import numpy as np
import random
from pathlib import Path
from math import log10, ceil

random.seed(4321)
np.random.seed(4321)
tf.set_random_seed(4321)

number_of_subjects = 1000

max_day = 90
patient_selection = 'ravvaz'
dose_history = 10
INR_history = 10

agents = {
    # 'WCA04_05_0.5': WarfarinClusterAgent(cluster_filename='Weka output (4 clusters).csv', smoothing_dose_threshold=5, dose_step=0.5),
    # 'WCA06_05_0.5': WarfarinClusterAgent(cluster_filename='Weka output (6 clusters).csv', smoothing_dose_threshold=5, dose_step=0.5),
    # 'WCA16_05_0.5': WarfarinClusterAgent(cluster_filename='Weka output (16 clusters).csv', smoothing_dose_threshold=5, dose_step=0.5)
    # 'WCA04_05_0.0': WarfarinClusterAgent(cluster_filename='Weka output (4 clusters).csv', smoothing_dose_threshold=5),
    # 'WCA06_05_0.0': WarfarinClusterAgent(cluster_filename='Weka output (6 clusters).csv', smoothing_dose_threshold=5),
    # 'WCA16_05_0.0': WarfarinClusterAgent(cluster_filename='Weka output (16 clusters).csv', smoothing_dose_threshold=5)
    # 'WCA04_00_0.0': WarfarinClusterAgent(cluster_filename='Weka output (4 clusters).csv'),
    # 'WCA06_00_0.0': WarfarinClusterAgent(cluster_filename='Weka output (6 clusters).csv'),
    # 'WCA16_00_0.0': WarfarinClusterAgent(cluster_filename='Weka output (16 clusters).csv'),
    # 'WCA04_00_0.5': WarfarinClusterAgent(cluster_filename='Weka output (4 clusters).csv', dose_step=0.5),
    # 'WCA06_00_0.5': WarfarinClusterAgent(cluster_filename='Weka output (6 clusters).csv', dose_step=0.5),
    # 'WCA16_00_0.5': WarfarinClusterAgent(cluster_filename='Weka output (16 clusters).csv', dose_step=0.5)
    '0.00': DQNAgent(path='./WARFV5d_90dose_10INR_10Tdose_change_coef_0.00DQN(32,32,32)g0.95efnbff900clrFbtch50vld0.3.data', filename='protocol')
    # '00.0_2_day': DQNAgent(path='./WARFV4_d_90_dose_9_INR_9_T__dose_change_coef_0.00_DQN_(20,20)_g_0.95_e_func_buff_900_clr_F_btch_50_vld_0.3.data', filename='protocol'),
    # '00.2_2_day': DQNAgent(path='./WARFV4_d_90_dose_9_INR_9_T__dose_change_coef_0.20_DQN_(20,20)_g_0.95_e_func_buff_900_clr_F_btch_50_vld_0.3.data', filename='protocol'),
    # '00.5_7_day': DQNAgent(path='./WARFV4_d_90_dose_9_INR_9_T__dose_change_coef_0.50_DQN_(20,20)_g_0.95_e_func_buff_900_clr_F_btch_50_vld_0.3.data', filename='protocol'),
    # '00.6_2_day': DQNAgent(path='./WARFV4_d_90_dose_9_INR_9_T__dose_change_coef_0.60_DQN_(20,20)_g_0.95_e_func_buff_900_clr_F_btch_50_vld_0.3.data', filename='protocol'),
    # '00.8_2_day': DQNAgent(path='./WARFV4_d_90_dose_9_INR_9_T__dose_change_coef_0.80_DQN_(20,20)_g_0.95_e_func_buff_900_clr_F_btch_50_vld_0.3.data', filename='protocol'),
    # '01.0_2_day': DQNAgent(path='./WARFV4_d_90_dose_9_INR_9_T__dose_change_coef_1.00_DQN_(20,20)_g_0.95_e_func_buff_900_clr_F_btch_50_vld_0.3.data', filename='protocol'),
    # '05.0_2_day': DQNAgent(path='./WARFV5_d_90_dose_9_INR_9_T__dose_change_coef_5.00_DQN_(20,20)_g_0.95_e_func_buff_900_clr_F_btch_50_vld_0.3.data', filename='protocol'),
    # '10.0_2_day': DQNAgent(path='./WARFV5_d_90_dose_9_INR_9_T__dose_change_coef_10.00_DQN_(20,20)_g_0.95_e_func_buff_900_clr_F_btch_50_vld_0.3.data', filename='protocol')
}

subjects = {'': WarfarinModel_v5(max_day=max_day,
                                            patient_selection=patient_selection,
                                            dose_history=dose_history,
                                            INR_history=INR_history,
                                            randomized=True,
                                            extended_state=False,
                                            save_patient=False,
                                            patient_save_prefix='')}

def index_generator(number_of_subjects, start_number=0, digits=6):
    for i in range(start_number, start_number + number_of_subjects):
        yield str(i).rjust(digits,'0')

exp = Experiment()

exp.add(agents=agents, subjects=subjects)

exp.generate_subjects(number_of_subjects=number_of_subjects, subjects_path='./patients', file_index_generator=index_generator)

for agent in agents:
    exp.assign([(agent, '')])
    print(agent)
    exp.run(subjects_path='./patients', file_index_generator=index_generator, number_of_subjects=number_of_subjects)
    exp.divest([(agent, '')])


# subject = subjects['Warfv5']
# digits = ceil(log10(number_of_subjects))
# for subject_ID in range(number_of_subjects):
#         filename = 'Warfv5' + str(subject_ID).rjust(digits, '0')
#         subject.load(filename=filename, path='./' + 'Warfv5')
#         subject._extended_state = True
#         subject.save(filename=filename, path='./' + 'Warfv5')


agents = {
    'AAA': WarfarinAgent(study_arm='AAA'),
    'CAA': WarfarinAgent(study_arm='CAA'),
    'PGAA': WarfarinAgent(study_arm='PGAA'),
#     'PGPGA': WarfarinAgent(study_arm='PGPGA')
}

exp = Experiment(number_of_subjects=number_of_subjects)

exp.add(agents=agents, subjects=subjects)
for agent in agents:
    exp.assign([(agent, 'Warfv5')])
    print(agent)
    exp.run()
    exp.divest([(agent, 'Warfv5')])
