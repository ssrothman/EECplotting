from datasets import get_pickled_histogram, get_unfolded_histogram, get_pickled_histogram_sum
import pickle
import plotters.res4
from importlib import reload
import numpy as np
import matplotlib.pyplot as plt

with open("/ceph/submit/data/group/cms/store/user/srothman/EEC/Apr_23_2025/Herwig_inclusive/EECres4tee/UNFOLDED/RECO_boot2000/LOSS_2d_Apr_23_2025_Pythia_inclusive_boot2000/Hunf_statonly.pkl", 'rb') as f:
    Hunf_herwig_2d_statonly = pickle.load(f)

with open("/ceph/submit/data/group/cms/store/user/srothman/EEC/Apr_23_2025/Herwig_inclusive/EECres4tee/UNFOLDED/RECO_boot2000/LOSS_2d_Apr_23_2025_Pythia_inclusive_boot2000/Hunf.pkl", 'rb') as f:
    Hunf_herwig_2d = pickle.load(f)

with open("/ceph/submit/data/group/cms/store/user/srothman/EEC/Apr_23_2025/Pythia_inclusive/EECres4tee/UNFOLDED/RECO_boot2000_2stat1/LOSS_2d_Apr_23_2025_Pythia_inclusive_boot2000_2stat0/Hunf.pkl", 'rb') as f:
    Hunf_pythia_2d = pickle.load(f)

with open("/ceph/submit/data/group/cms/store/user/srothman/EEC/Apr_23_2025/Pythia_inclusive/EECres4tee/UNFOLDED/RECO_boot2000_2stat1/LOSS_2d_Apr_23_2025_Pythia_inclusive_boot2000_2stat0/Hunf_statonly.pkl", 'rb') as f:
    Hunf_pythia_2d_statonly = pickle.load(f)

with open("/ceph/submit/data/group/cms/store/user/srothman/EEC/Apr_23_2025/Pythia_inclusive/EECres4tee/UNFOLDED/RECO_boot2000_2stat1/LOSS_2d_Apr_23_2025_Pythia_inclusive_boot2000_2stat0/minimization_result.pkl", 'rb') as f:
    res_pythia = pickle.load(f)[0]

with open("/ceph/submit/data/group/cms/store/user/srothman/EEC/Apr_23_2025/Herwig_inclusive/EECres4tee/UNFOLDED/RECO_boot2000/LOSS_2d_Apr_23_2025_Pythia_inclusive_boot2000/minimization_result.pkl", 'rb') as f:
    res_herwig = pickle.load(f)[0]
 


Hgen_herwig = get_pickled_histogram('Apr_23_2025', 'Herwig_inclusive', 
                                    'EECres4tee', 'nominal', 'nominal',
                                    'gen', max_nboot=2000)

Hgen_pythia = get_pickled_histogram('Apr_23_2025', 'Herwig_inclusive', 
                                    'EECres4tee', 'nominal', 'nominal',
                                    'gen', max_nboot=2000,
                                    statN = 2, statK = 0)
