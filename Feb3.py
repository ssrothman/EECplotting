import pickle

tee_path = '/ceph/submit/data/user/s/srothman/EEC/Jan_22_2024/Data_2018D/EECres4tee/hists_file0to100_poissonbootstrap1000_noSyst_data.pkl'
dipole_path = '/ceph/submit/data/user/s/srothman/EEC/Jan_22_2024/Data_2018D/EECres4dipole/hists_file0to100_poissonbootstrap1000_noSyst_data.pkl'
triangle_path = '/ceph/submit/data/user/s/srothman/EEC/Jan_22_2024/Data_2018D/EECres4triangle/hists_file0to563_poissonbootstrap1000_noSyst_data.pkl'

tee_MC_path = '/ceph/submit/data/user/s/srothman/EEC/Dec_26_2024/Herwig_glu/EECres4tee/hists_file0to1047_poissonbootstrap1000_noSyst_genonly.pkl'
dipole_MC_path = '/ceph/submit/data/user/s/srothman/EEC/Dec_26_2024/Herwig_glu/EECres4dipole/hists_file0to1047_poissonbootstrap1000_noSyst_genonly.pkl'
triangle_MC_path = '/ceph/submit/data/user/s/srothman/EEC/Dec_26_2024/Herwig_glu/EECres4triangle/hists_file0to1047_poissonbootstrap1000_noSyst_genonly.pkl'


with open(tee_path, 'rb') as f:
    tee_data = pickle.load(f)['nominal']['reco']

with open(dipole_path, 'rb') as f:
    dipole_data = pickle.load(f)['nominal']['reco']

with open(triangle_path, 'rb') as f:
    triangle_data = pickle.load(f)['nominal']['reco']

with open(tee_MC_path, 'rb') as f:
    tee_MC_data = pickle.load(f)['nominal']['reco']

with open(dipole_MC_path, 'rb') as f:
    dipole_MC_data = pickle.load(f)['nominal']['reco']

with open(triangle_MC_path, 'rb') as f:
    triangle_MC_data = pickle.load(f)['nominal']['reco']

import res4
from importlib import reload
import matplotlib.pyplot as plt
import numpy as np

