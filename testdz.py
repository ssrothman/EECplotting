import pickle

with open("/data/submit/srothman/EEC/Jun28_2024/DYJetsToLL_nodz/EECproj/hists_file0to4927_tight.pkl", 'rb') as f:
    nodz = pickle.load(f)

with open("/data/submit/srothman/EEC/Jun28_2024/DYJetsToLL/EECproj/hists_file0to6276_tight_manualcov.pkl", 'rb') as f:
    default = pickle.load(f)

from proj import *

compareProjectedEEC(default['reco'], default['covreco'],
                    nodz['reco'], nodz['covreco'],
                    binning1 = {'order':0,
                                'btag':0,
                                'pt' : 2},
                    binning2 = {'order':0,
                                'btag':0,
                                'pt' : 2},
                    label1='default',
                    label2='nodz')

compareProjectedEEC(default['reco'], default['covreco'],
                    nodz['reco'], nodz['covreco'],
                    binning1 = {'order':0,
                                'btag':1,
                                'pt' : 2},
                    binning2 = {'order':0,
                                'btag':1,
                                'pt' : 2},
                    label1='default',
                    label2='nodz')
