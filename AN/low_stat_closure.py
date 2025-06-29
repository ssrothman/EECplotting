from datasets import get_pickled_histogram, get_unfolded_histogram, get_pickled_histogram_sum
import pickle
import plotters.res4
from importlib import reload
import numpy as np
import os
import matplotlib.pyplot as plt

with open("/ceph/submit/data/group/cms/store/user/srothman/EEC/Apr_23_2025/Herwig_inclusive/EECres4tee/UNFOLDED/RECO_boot-1/LOSS_1d_Apr_23_2025_Pythia_inclusive_boot-1/Hunf.pkl", 'rb') as f:
    Hunf_herwig = pickle.load(f)

with open("/ceph/submit/data/group/cms/store/user/srothman/EEC/Apr_23_2025/Herwig_inclusive/EECres4tee/UNFOLDED/RECO_boot-1/LOSS_1d_Apr_23_2025_Pythia_inclusive_boot-1/Hunf_statonly.pkl", 'rb') as f:
    Hunf_herwig_statonly = pickle.load(f)

with open("/ceph/submit/data/group/cms/store/user/srothman/EEC/Apr_23_2025/Pythia_inclusive/EECres4tee/UNFOLDED/RECO_boot-1_2stat1/LOSS_1d_Apr_23_2025_Pythia_inclusive_boot-1_2stat0/Hunf.pkl", 'rb') as f:
    Hunf_Spythia = pickle.load(f)

with open("/ceph/submit/data/group/cms/store/user/srothman/EEC/Apr_23_2025/Pythia_inclusive/EECres4tee/UNFOLDED/RECO_boot-1_2stat1/LOSS_1d_Apr_23_2025_Pythia_inclusive_boot-1_2stat0/Hunf_statonly.pkl", 'rb') as f:
    Hunf_Spythia_statonly = pickle.load(f)

with open("/ceph/submit/data/group/cms/store/user/srothman/EEC/Apr_23_2025/Pythia_inclusive/EECres4tee/UNFOLDED/RECO_boot-1/LOSS_1d_Apr_23_2025_Pythia_inclusive_boot-1/Hunf.pkl", 'rb') as f:
     Hunf_pythia = pickle.load(f)

with open("/ceph/submit/data/group/cms/store/user/srothman/EEC/Apr_23_2025/Pythia_inclusive/EECres4tee/UNFOLDED/RECO_boot-1/LOSS_1d_Apr_23_2025_Pythia_inclusive_boot-1/Hunf_statonly.pkl", 'rb') as f:
    Hunf_pythia_statonly = pickle.load(f)

with open("/ceph/submit/data/group/cms/store/user/srothman/EEC/Apr_23_2025/Herwig_inclusive/EECres4tee/UNFOLDED/RECO_boot-1/LOSS_1d_Apr_23_2025_Pythia_inclusive_boot-1/minimization_result.pkl", 'rb') as f:
    res_herwig = pickle.load(f)[0]

with open("/ceph/submit/data/group/cms/store/user/srothman/EEC/Apr_23_2025/Pythia_inclusive/EECres4tee/UNFOLDED/RECO_boot-1_2stat1/LOSS_1d_Apr_23_2025_Pythia_inclusive_boot-1_2stat0/minimization_result.pkl", 'rb') as f:
    res_Spythia = pickle.load(f)[0]
 
with open("/ceph/submit/data/group/cms/store/user/srothman/EEC/Apr_23_2025/Pythia_inclusive/EECres4tee/UNFOLDED/RECO_boot-1/LOSS_1d_Apr_23_2025_Pythia_inclusive_boot-1/minimization_result.pkl", 'rb') as f:
    res_pythia = pickle.load(f)[0]

Hgen_herwig = get_pickled_histogram('Apr_23_2025', 'Herwig_inclusive', 
                                    'EECres4tee', 'nominal', 'nominal',
                                    'gen',
                                    reweight=None)

Hgen_Spythia = get_pickled_histogram('Apr_23_2025', 'Pythia_inclusive', 
                                    'EECres4tee', 'nominal', 'nominal',
                                    'gen',
                                    statN = 2, statK = 1,
                                     reweight=None)

Hgen_pythia = get_pickled_histogram('Apr_23_2025', 'Pythia_inclusive', 
                                    'EECres4tee', 'nominal', 'nominal',
                                    'gen',
                                    reweight=None)

os.makedirs('AN/low_stat_closure_1d', exist_ok=True)
os.makedirs('AN/low_stat_closure_1d/perfect', exist_ok=True)
os.makedirs('AN/low_stat_closure_1d/stat', exist_ok=True)
os.makedirs('AN/low_stat_closure_1d/model', exist_ok=True)

plotters.res4.plot_correlations(res_herwig.invhess, 
                                savefig='AN/low_stat_closure_1d/model/correlations.png')
plotters.res4.plot_correlations(res_Spythia.invhess,
                                savefig='AN/low_stat_closure_1d/stat/correlations.png')
plotters.res4.plot_correlations(res_pythia.invhess,
                                savefig='AN/low_stat_closure_1d/perfect/correlations.png')

plotters.res4.plot_correlations(res_herwig.invhess[:7875, :7875], 
                                savefig='AN/low_stat_closure_1d/model/correlations_zoom.png')
plotters.res4.plot_correlations(res_Spythia.invhess[:7875, :7875],
                                savefig='AN/low_stat_closure_1d/stat/correlations_zoom.png')
plotters.res4.plot_correlations(res_pythia.invhess[:7875, :7875],
                                savefig='AN/low_stat_closure_1d/perfect/correlations_zoom.png')

plotters.res4.plot_pulls(res_herwig, 
                         savefig='AN/low_stat_closure_1d/model/pulls.png')
plotters.res4.plot_pulls(res_Spythia,
                         savefig='AN/low_stat_closure_1d/stat/pulls.png')
plotters.res4.plot_pulls(res_pythia,
                         savefig='AN/low_stat_closure_1d/perfect/pulls.png')

plotters.res4.plot_named_pulls(res_herwig,
                               savefig='AN/low_stat_closure_1d/model/named_pulls.png')
plotters.res4.plot_named_pulls(res_Spythia,
                               savefig='AN/low_stat_closure_1d/stat/named_pulls.png')
plotters.res4.plot_named_pulls(res_pythia,
                               savefig='AN/low_stat_closure_1d/perfect/named_pulls.png')

for ptbin in range(5):
    for rcbin in [0, 5, 10]:
        plotters.res4.compare_1d(Hgen_herwig, 
                                 Hunf_herwig, 
                                 "Gen", 
                                 "Unfolded [stat+syst]", 
                                 1, ptbin, r=rcbin, c=rcbin,
                                 savefig='AN/low_stat_closure_1d/model/UNF')

        plotters.res4.compare_1d(Hgen_Spythia, 
                                 Hunf_Spythia_statonly, 
                                 "Gen", 
                                 "Unfolded [stat only]", 
                                 1, ptbin, r=rcbin, c=rcbin,
                                 savefig='AN/low_stat_closure_1d/stat/UNF')

        plotters.res4.compare_1d(Hgen_pythia, 
                                 Hunf_pythia_statonly, 
                                 "Gen", 
                                 "Unfolded [stat only]", 
                                 1, ptbin, r=rcbin, c=rcbin,
                                 savefig='AN/low_stat_closure_1d/perfect/UNF')
