import pickle

hist_paths = {
    "Pythia_glu" : {
        "res4tee": "/ceph/submit/data/user/s/srothman/EEC/Dec_26_2024/Pythia_glu/EECres4tee/hists_file0to3717_poissonbootstrap1000_noSyst_genonly.pkl",
        "res4dipole": "/ceph/submit/data/user/s/srothman/EEC/Dec_26_2024/Pythia_glu/EECres4dipole/hists_file0to3717_poissonbootstrap1000_noSyst_genonly.pkl",
        "res4triangle": "/ceph/submit/data/user/s/srothman/EEC/Dec_26_2024/Pythia_glu/EECres4triangle/hists_file0to3717_poissonbootstrap1000_noSyst_genonly.pkl"
    },
    "Pythia_q" : {
        "res4tee": "/ceph/submit/data/user/s/srothman/EEC/Dec_26_2024/Pythia_q/EECres4tee/hists_file0to2987_poissonbootstrap1000_noSyst_genonly.pkl",
        "res4dipole": "/ceph/submit/data/user/s/srothman/EEC/Dec_26_2024/Pythia_q/EECres4dipole/hists_file0to2987_poissonbootstrap1000_noSyst_genonly.pkl",
        "res4triangle": "/ceph/submit/data/user/s/srothman/EEC/Dec_26_2024/Pythia_q/EECres4triangle/hists_file0to2987_poissonbootstrap1000_noSyst_genonly.pkl"
    },
    "Pythia_glu_nospin" : {
        #"res4tee": "/ceph/submit/data/user/s/srothman/EEC/Dec_26_2024/Pythia_glu_nospin/EECres4tee/hists_file0to2289_poissonbootstrap1000_noSyst_genonly.pkl",
        #"res4dipole": "/ceph/submit/data/user/s/srothman/EEC/Dec_26_2024/Pythia_glu_nospin/EECres4dipole/hists_file0to2289_poissonbootstrap1000_noSyst_genonly.pkl",
        #"res4triangle": "/ceph/submit/data/user/s/srothman/EEC/Dec_26_2024/Pythia_glu_nospin/EECres4triangle/hists_file0to2289_poissonbootstrap1000_noSyst_genonly.pkl"
    },
    "Pythia_q_nospin" : {
        "res4tee": "/ceph/submit/data/user/s/srothman/EEC/Dec_26_2024/Pythia_q_nospin/EECres4tee/hists_file0to1035_poissonbootstrap1000_noSyst_genonly.pkl",
        "res4dipole": "/ceph/submit/data/user/s/srothman/EEC/Dec_26_2024/Pythia_q_nospin/EECres4dipole/hists_file0to1035_poissonbootstrap1000_noSyst_genonly.pkl",
        "res4triangle": "/ceph/submit/data/user/s/srothman/EEC/Dec_26_2024/Pythia_q_nospin/EECres4triangle/hists_file0to1035_poissonbootstrap1000_noSyst_genonly.pkl"
    },
    "Herwig_glu" : {
        "res4tee": "/ceph/submit/data/user/s/srothman/EEC/Dec_26_2024/Herwig_glu/EECres4tee/hists_file0to1047_poissonbootstrap1000_noSyst_genonly.pkl",
        "res4dipole": "/ceph/submit/data/user/s/srothman/EEC/Dec_26_2024/Herwig_glu/EECres4dipole/hists_file0to1047_poissonbootstrap1000_noSyst_genonly.pkl",
        "res4triangle": "/ceph/submit/data/user/s/srothman/EEC/Dec_26_2024/Herwig_glu/EECres4triangle/hists_file0to1047_poissonbootstrap1000_noSyst_genonly.pkl",
    },
    "Herwig_glu_gg" : {
        "res4tee" : "/ceph/submit/data/user/s/srothman/EEC/Dec_26_2024/Herwig_glu_gg/EECres4tee/hists_file0to1037_poissonbootstrap1000_noSyst_genonly.pkl",
        "res4dipole" : "/ceph/submit/data/user/s/srothman/EEC/Dec_26_2024/Herwig_glu_gg/EECres4dipole/hists_file0to1037_poissonbootstrap1000_noSyst_genonly.pkl",
        "res4triangle" : "/ceph/submit/data/user/s/srothman/EEC/Dec_26_2024/Herwig_glu_gg/EECres4triangle/hists_file0to1037_poissonbootstrap1000_noSyst_genonly.pkl",
    },
    "Herwig_glu_nospin" : {
        "res4tee" : "/ceph/submit/data/user/s/srothman/EEC/Dec_26_2024/Herwig_glu_nospin/EECres4tee/hists_file0to998_poissonbootstrap1000_noSyst_genonly.pkl",
        "res4dipole" : "/ceph/submit/data/user/s/srothman/EEC/Dec_26_2024/Herwig_glu_nospin/EECres4dipole/hists_file0to998_poissonbootstrap1000_noSyst_genonly.pkl",
        "res4triangle" : "/ceph/submit/data/user/s/srothman/EEC/Dec_26_2024/Herwig_glu_nospin/EECres4triangle/hists_file0to998_poissonbootstrap1000_noSyst_genonly.pkl"
    },
    "Herwig_glu_nospin_gg" : {
        "res4tee" : "/ceph/submit/data/user/s/srothman/EEC/Dec_26_2024/Herwig_glu_nospin_gg/EECres4tee/hists_file0to973_poissonbootstrap1000_noSyst_genonly.pkl",
        "res4dipole" : "/ceph/submit/data/user/s/srothman/EEC/Dec_26_2024/Herwig_glu_nospin_gg/EECres4dipole/hists_file0to973_poissonbootstrap1000_noSyst_genonly.pkl",
        "res4triangle" : "/ceph/submit/data/user/s/srothman/EEC/Dec_26_2024/Herwig_glu_nospin_gg/EECres4triangle/hists_file0to973_poissonbootstrap1000_noSyst_genonly.pkl"
    },
    "Herwig_q" : {
        "res4tee" : "/ceph/submit/data/user/s/srothman/EEC/Dec_26_2024/Herwig_q/EECres4tee/hists_file0to973_poissonbootstrap1000_noSyst_genonly.pkl",
        "res4dipole" : "/ceph/submit/data/user/s/srothman/EEC/Dec_26_2024/Herwig_q/EECres4dipole/hists_file0to973_poissonbootstrap1000_noSyst_genonly.pkl",
        "res4triangle" : "/ceph/submit/data/user/s/srothman/EEC/Dec_26_2024/Herwig_q/EECres4triangle/hists_file0to973_poissonbootstrap1000_noSyst_genonly.pkl",
    },
    "Herwig_q_nogg" : {
        "res4tee" : "/ceph/submit/data/user/s/srothman/EEC/Dec_26_2024/Herwig_q_nogg/EECres4tee/hists_file0to937_poissonbootstrap1000_noSyst_genonly.pkl",
        "res4dipole" : "/ceph/submit/data/user/s/srothman/EEC/Dec_26_2024/Herwig_q_nogg/EECres4dipole/hists_file0to937_poissonbootstrap1000_noSyst_genonly.pkl",
        "res4triangle" : "/ceph/submit/data/user/s/srothman/EEC/Dec_26_2024/Herwig_q_nogg/EECres4triangle/hists_file0to937_poissonbootstrap1000_noSyst_genonly.pkl"
    },
    "Herwig_q_nospin" : {
        "res4tee" : "/ceph/submit/data/user/s/srothman/EEC/Dec_26_2024/Herwig_q_nospin/EECres4tee/hists_file0to905_poissonbootstrap1000_noSyst_genonly.pkl",
        "res4dipole" : "/ceph/submit/data/user/s/srothman/EEC/Dec_26_2024/Herwig_q_nospin/EECres4dipole/hists_file0to905_poissonbootstrap1000_noSyst_genonly.pkl",
        "res4triangle" : "/ceph/submit/data/user/s/srothman/EEC/Dec_26_2024/Herwig_q_nospin/EECres4triangle/hists_file0to905_poissonbootstrap1000_noSyst_genonly.pkl",
    },
    "Herwig_q_nospin_nogg" : {
        "res4tee" : "/ceph/submit/data/user/s/srothman/EEC/Dec_26_2024/Herwig_q_nospin_nogg/EECres4tee/hists_file0to866_poissonbootstrap1000_noSyst_genonly.pkl",
        "res4dipole" : "/ceph/submit/data/user/s/srothman/EEC/Dec_26_2024/Herwig_q_nospin_nogg/EECres4dipole/hists_file0to866_poissonbootstrap1000_noSyst_genonly.pkl",
        "res4triangle" : "/ceph/submit/data/user/s/srothman/EEC/Dec_26_2024/Herwig_q_nospin_nogg/EECres4triangle/hists_file0to866_poissonbootstrap1000_noSyst_genonly.pkl",
    }
}

names = {
    "Pythia_glu" : "Pythia Z+glu",
    "Pythia_q" : "Pythia Z+q",
    "Pythia_glu_nospin" : "Pythia Z+glu (no spin corr)",
    "Pythia_q_nospin" : "Pythia Z+q (no spin corr)",
    "Herwig_glu" : "Herwig Z+glu",
    "Herwig_glu_gg" : "Herwig Z+glu (no g->qq)",
    "Herwig_glu_nospin" : "Herwig Z+glu (no spin corr)",
    "Herwig_glu_nospin_gg" : "Herwig Z+glu (no g->qq, no spin corr)",
    "Herwig_q" : "Herwig Z+q",
    "Herwig_q_nogg" : "Herwig Z+q (no g->gg)",
    "Herwig_q_nospin" : "Herwig Z+q (no spin corr)",
    "Herwig_q_nospin_nogg" : "Herwig Z+q (no g->gg, no spin corr)"
}

bad = [
    "/ceph/submit/data/user/s/srothman/EEC/Dec_26_2024/Pythia_glu_nospin/EECres4tee/hists_file0to2289_poissonbootstrap1000_noSyst_genonly.pkl",
    "/ceph/submit/data/user/s/srothman/EEC/Dec_26_2024/Pythia_glu_nospin/EECres4dipole/hists_file0to2289_poissonbootstrap1000_noSyst_genonly.pkl",
    "/ceph/submit/data/user/s/srothman/EEC/Dec_26_2024/Pythia_glu_nospin/EECres4triangle/hists_file0to2289_poissonbootstrap1000_noSyst_genonly.pkl",
]

res4tee_hists = {}
for key in hist_paths:
    if 'res4tee' in hist_paths[key]:
        print(hist_paths[key]['res4tee'])
        with open(hist_paths[key]['res4tee'], 'rb') as f:
            res4tee_hists[key] = pickle.load(f)['nominal']['reco']

res4dipole_hists = {}
for key in hist_paths:
    if 'res4dipole' in hist_paths[key]:
        print(hist_paths[key]['res4dipole'])
        with open(hist_paths[key]['res4dipole'], 'rb') as f:
            res4dipole_hists[key] = pickle.load(f)['nominal']['reco']

res4triangle_hists = {}
for key in hist_paths:
    if 'res4triangle' in hist_paths[key]:
        print(hist_paths[key]['res4triangle'])
        with open(hist_paths[key]['res4triangle'], 'rb') as f:
            res4triangle_hists[key] = pickle.load(f)['nominal']['reco']

#res3_hists = {}
#for key in hist_paths:
#    if 'res3' in hist_paths[key]:
#        with open(hist_paths[key]['res3'], 'rb') as f:
#            res3_hists[key] = pickle.load(f)['nominal']['reco']

#proj_hists = {}
#for key in hist_paths:
#    if 'proj' in hist_paths[key]:
#        with open(hist_paths[key]['proj'], 'rb') as f:
#            proj_hists[key] = pickle.load(f)['nominal']['reco']

import res4
#import res3
#import proj

from importlib import reload
import matplotlib.pyplot as plt
import numpy as np
from util import savefig

#for key in hist_paths:
#    for ptbin in range(5):
#        for RLbin in range(1, 9):
#            res4.plot_total_heatmap(res4tee_hists[key], 
#                                    title=names[key],
#                                    btag=0, ptbin=ptbin, RLbin=RLbin,
#                                    triangle=False,
#                                    show=False)
#            savefig("/ceph/submit/data/user/s/srothman/EEC/Dec_09_2024/plots/TEE_total_heatmap_%s_ptbin%d_RLbin%d.png" % (key, ptbin, RLbin))
#            plt.close()
#
#            res4.plot_modulation_heatmap(res4tee_hists[key], 
#                                    title=names[key],
#                                    btag=0, ptbin=ptbin, RLbin=RLbin,
#                                    show=False)
#            savefig("/ceph/submit/data/user/s/srothman/EEC/Dec_09_2024/plots/TEE_modulation_heatmap_%s_ptbin%d_RLbin%d.png" % (key, ptbin, RLbin))
#            plt.close()
#
#            res4.plot_modulation_heatmap(res4tee_hists[key], 
#                                    title=names[key],
#                                    btag=0, ptbin=ptbin, RLbin=RLbin,
#                                    show=False)
#            savefig("/ceph/submit/data/user/s/srothman/EEC/Dec_09_2024/plots/TEE_modulation_heatmap_%s_ptbin%d_RLbin%d.png" % (key, ptbin, RLbin))
#            plt.close()
#
#            res4.plot_profiles(res4tee_hists[key], 
#                               title=names[key],
#                               btag=0, ptbin=ptbin, RLbin=RLbin,
#                               show=False)
#            savefig("/ceph/submit/data/user/s/srothman/EEC/Dec_09_2024/plots/TEE_profiles_%s_ptbin%d_RLbin%d.png" % (key, ptbin, RLbin))
#            plt.close()
#
#            res4.plot_total_heatmap(res4dipole_hists[key],
#                                    title=names[key],
#                                    btag=0, ptbin=ptbin, RLbin=RLbin,
#                                    triangle=False,
#                                    show=False)
#            savefig("/ceph/submit/data/user/s/srothman/EEC/Dec_09_2024/plots/DIPOLE_total_heatmap_%s_ptbin%d_RLbin%d.png" % (key, ptbin, RLbin))
#            plt.close()
#
#            res4.plot_total_heatmap(res4triangle_hists[key],
#                                    title=names[key],
#                                    btag=0, ptbin=ptbin, RLbin=RLbin,
#                                    triangle=True,
#                                    show=False)
#            savefig("/ceph/submit/data/user/s/srothman/EEC/Dec_09_2024/plots/TRIANGLE_total_heatmap_%s_ptbin%d_RLbin%d.png" % (key, ptbin, RLbin))
#            plt.close()
