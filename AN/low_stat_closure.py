from datasets import get_pickled_histogram, get_unfolded_histogram, get_pickled_histogram_sum
import pickle
import plotters.res4
from importlib import reload
import numpy as np
import os
import matplotlib.pyplot as plt

with open("/ceph/submit/data/group/cms/store/user/srothman/EEC/Apr_23_2025/Herwig_inclusive/EECres4tee/UNFOLDED/RECO_boot2000/LOSS_2d_Apr_23_2025_Pythia_inclusive_boot2000/Hunf.pkl", 'rb') as f:
    Hunf_herwig = pickle.load(f)

with open("/ceph/submit/data/group/cms/store/user/srothman/EEC/Apr_23_2025/Herwig_inclusive/EECres4tee/UNFOLDED/RECO_boot2000/LOSS_2d_Apr_23_2025_Pythia_inclusive_boot2000/Hunf_statonly.pkl", 'rb') as f:
    Hunf_herwig_statonly = pickle.load(f)

with open("/ceph/submit/data/group/cms/store/user/srothman/EEC/Apr_23_2025/Pythia_inclusive/EECres4tee/UNFOLDED/RECO_boot2000_2stat1/LOSS_2d_Apr_23_2025_Pythia_inclusive_boot2000_2stat0/Hunf.pkl", 'rb') as f:
    Hunf_Spythia = pickle.load(f)

with open("/ceph/submit/data/group/cms/store/user/srothman/EEC/Apr_23_2025/Pythia_inclusive/EECres4tee/UNFOLDED/RECO_boot2000_2stat1/LOSS_2d_Apr_23_2025_Pythia_inclusive_boot2000_2stat0/Hunf_statonly.pkl", 'rb') as f:
    Hunf_Spythia_statonly = pickle.load(f)

with open("/ceph/submit/data/group/cms/store/user/srothman/EEC/Apr_23_2025/Pythia_inclusive/EECres4tee/UNFOLDED/RECO_boot2000/LOSS_2d_Apr_23_2025_Pythia_inclusive_boot2000/Hunf.pkl", 'rb') as f:
     Hunf_pythia = pickle.load(f)

with open("/ceph/submit/data/group/cms/store/user/srothman/EEC/Apr_23_2025/Pythia_inclusive/EECres4tee/UNFOLDED/RECO_boot2000/LOSS_2d_Apr_23_2025_Pythia_inclusive_boot2000/Hunf_statonly.pkl", 'rb') as f:
    Hunf_pythia_statonly = pickle.load(f)

with open("/ceph/submit/data/group/cms/store/user/srothman/EEC/Apr_23_2025/Herwig_inclusive/EECres4tee/UNFOLDED/RECO_boot2000/LOSS_2d_Apr_23_2025_Pythia_inclusive_boot2000/minimization_result.pkl", 'rb') as f:
    res_herwig = pickle.load(f)[0]

with open("/ceph/submit/data/group/cms/store/user/srothman/EEC/Apr_23_2025/Pythia_inclusive/EECres4tee/UNFOLDED/RECO_boot2000_2stat1/LOSS_2d_Apr_23_2025_Pythia_inclusive_boot2000_2stat0/minimization_result.pkl", 'rb') as f:
    res_Spythia = pickle.load(f)[0]
 
with open("/ceph/submit/data/group/cms/store/user/srothman/EEC/Apr_23_2025/Pythia_inclusive/EECres4tee/UNFOLDED/RECO_boot2000/LOSS_2d_Apr_23_2025_Pythia_inclusive_boot2000/minimization_result.pkl", 'rb') as f:
    res_pythia = pickle.load(f)[0]

Hgen_herwig = get_pickled_histogram('Apr_23_2025', 'Herwig_inclusive', 
                                    'EECres4tee', 'nominal', 'nominal',
                                    'gen', max_nboot=2000)

Hgen_Spythia = get_pickled_histogram('Apr_23_2025', 'Herwig_inclusive', 
                                    'EECres4tee', 'nominal', 'nominal',
                                    'gen', max_nboot=2000,
                                    statN = 2, statK = 1)

Hgen_pythia = get_pickled_histogram('Apr_23_2025', 'Herwig_inclusive', 
                                    'EECres4tee', 'nominal', 'nominal',
                                    'gen', max_nboot=2000)

os.makedirs('AN/low_stat_closure', exist_ok=True)
os.makedirs('AN/low_stat_closure/perfect', exist_ok=True)
os.makedirs('AN/low_stat_closure/stat', exist_ok=True)
os.makedirs('AN/low_stat_closure/model', exist_ok=True)

plotters.res4.plot_correlations(res_herwig.invhess, 
                                savefig='AN/low_stat_closure/model/correlations.png')
plotters.res4.plot_correlations(res_Spythia.invhess,
                                savefig='AN/low_stat_closure/stat/correlations.png')
plotters.res4.plot_correlations(res_pythia.invhess,
                                savefig='AN/low_stat_closure/perfect/correlations.png')

plotters.res4.plot_pulls(res_herwig, 
                         savefig='AN/low_stat_closure/model/pulls.png')
plotters.res4.plot_pulls(res_Spythia,
                         savefig='AN/low_stat_closure/stat/pulls.png')
plotters.res4.plot_pulls(res_pythia,
                         savefig='AN/low_stat_closure/perfect/pulls.png')

plotters.res4.plot_named_pulls(res_herwig,
                               savefig='AN/low_stat_closure/model/named_pulls.png')
plotters.res4.plot_named_pulls(res_Spythia,
                               savefig='AN/low_stat_closure/stat/named_pulls.png')
plotters.res4.plot_named_pulls(res_pythia,
                               savefig='AN/low_stat_closure/perfect/named_pulls.png')

for ptbin in range(5):
    for rcbin in [0, 5, 10]:
        plotters.res4.compare_1d(Hgen_herwig, 
                                 Hunf_herwig, 
                                 "Gen", 
                                 "Unfolded [stat+syst]", 
                                 1, ptbin, r=rcbin, c=rcbin,
                                 savefig='AN/low_stat_closure/model/UNF')

        plotters.res4.compare_1d(Hgen_Spythia, 
                                 Hunf_Spythia_statonly, 
                                 "Gen", 
                                 "Unfolded [stat only]", 
                                 1, ptbin, r=rcbin, c=rcbin,
                                 savefig='AN/low_stat_closure/stat/UNF')

        plotters.res4.compare_1d(Hgen_pythia, 
                                 Hunf_pythia_statonly, 
                                 "Gen", 
                                 "Unfolded [stat only]", 
                                 1, ptbin, r=rcbin, c=rcbin,
                                 savefig='AN/low_stat_closure/perfect/UNF')
