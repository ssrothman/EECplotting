from proj import *
import pickle

samples = {
    "nominal" : "/ceph/submit/data/group/cms/store/user/srothman/condor/nominal/merged.pkl",
    "BtagEff" : "/ceph/submit/data/group/cms/store/user/srothman/condor/BtagEff/merged.pkl",
    "BtagSF" : "/ceph/submit/data/group/cms/store/user/srothman/condor/BtagSF/merged.pkl",
    "CBxsec" : "/ceph/submit/data/group/cms/store/user/srothman/condor/CBxsec/merged.pkl",
    "JetMET" : "/ceph/submit/data/group/cms/store/user/srothman/condor/JetMET/merged.pkl",
    "Lxsec" : "/ceph/submit/data/group/cms/store/user/srothman/condor/Lxsec/merged.pkl",
    "Muon" : "/ceph/submit/data/group/cms/store/user/srothman/condor/Muon/merged.pkl",
    "Pileup" : "/ceph/submit/data/group/cms/store/user/srothman/condor/Pileup/merged.pkl",
    "PS" : "/ceph/submit/data/group/cms/store/user/srothman/condor/PS/merged.pkl",
    "Theory" : "/ceph/submit/data/group/cms/store/user/srothman/condor/Theory/merged.pkl",
    "Trigger" : "/ceph/submit/data/group/cms/store/user/srothman/condor/Trigger/merged.pkl"
}

#hists = {}
#for syst, path in samples.items():
#    with open(path, "rb") as f:
#        hists[syst] = pickle.load(f)

#for samplekey in samples:
#    print(samplekey+":")
#    with open(samples[samplekey], 'rb') as f:
#        thehist = pickle.load(f)
#
#    x_l = []
#    covx_l = []
#    label_l = []
#    color_l = []
#
#    for key in thehist.keys():
#        if key == 'config':
#            continue
#        print("\t",key)
#
#        x_l.append(thehist[key]['reco'])
#        covx_l.append(thehist[key]['covreco'])
#        label_l.append(key)
#        color_l.append(None)
#
#    binning = {
#        'order' : 0,
#        'pt' : 3,
#        'btag' : 0
#    }
#    compareProjectedEEC(
#        x_l, covx_l,
#        label_l = label_l,
#        color_l = color_l,
#        binning_l = binning,
#        folder='systplots/%s'%samplekey)
#
#    binning = {
#        'order' : 0,
#        'pt' : 3,
#        'btag' : 1
#    }
#    compareProjectedEEC(
#        x_l, covx_l,
#        label_l = label_l,
#        color_l = color_l,
#        binning_l = binning,
#        folder='systplots/%s'%samplekey)
