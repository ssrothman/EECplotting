import proj
#import res3
import res4

import pickle
import matplotlib.pyplot as plt

paths = {
    'Herwig' : {
        'proj' : '/ceph/submit/data/user/s/srothman/EEC/Oct14_2024/Herwig/EECproj/hists_file0to241_poissonbootstrap1000_noSyst.pkl',
        'res3' : '/ceph/submit/data/user/s/srothman/EEC/Oct14_2024/Herwig/EECres3/hists_file0to241_poissonbootstrap1000_noSyst.pkl',
        'res4tee' : '/ceph/submit/data/user/s/srothman/EEC/Oct14_2024/Herwig/EECres4tee/hists_file0to241_poissonbootstrap1000_noSyst.pkl',
        'res4dipole' : '/ceph/submit/data/user/s/srothman/EEC/Oct14_2024/Herwig/EECres4dipole/hists_file0to241_poissonbootstrap1000_noSyst.pkl'
    },
    'Herwig_nospin' : {
        'proj' : '/ceph/submit/data/user/s/srothman/EEC/Oct14_2024/Herwig_nospin/EECproj/hists_file0to198_poissonbootstrap1000_noSyst.pkl',
        'res3' : '/ceph/submit/data/user/s/srothman/EEC/Oct14_2024/Herwig_nospin/EECres3/hists_file0to198_poissonbootstrap1000_noSyst.pkl',
        'res4tee' : '/ceph/submit/data/user/s/srothman/EEC/Oct14_2024/Herwig_nospin/EECres4tee/hists_file0to198_poissonbootstrap1000_noSyst.pkl',
        'res4dipole' : '/ceph/submit/data/user/s/srothman/EEC/Oct14_2024/Herwig_nospin/EECres4dipole/hists_file0to198_poissonbootstrap1000_noSyst.pkl'
    },
    'Herwig_nobdecay' : {
        'proj' : '/ceph/submit/data/user/s/srothman/EEC/Oct14_2024/Herwig_nobdecay/EECproj/hists_file0to192_poissonbootstrap1000_noSyst.pkl',
        'res3' : '/ceph/submit/data/user/s/srothman/EEC/Oct14_2024/Herwig_nobdecay/EECres3/hists_file0to192_poissonbootstrap1000_noSyst.pkl',
        'res4tee' : '/ceph/submit/data/user/s/srothman/EEC/Oct14_2024/Herwig_nobdecay/EECres4tee/hists_file0to192_poissonbootstrap1000_noSyst.pkl',
        'res4dipole' : '/ceph/submit/data/user/s/srothman/EEC/Oct14_2024/Herwig_nobdecay/EECres4dipole/hists_file0to192_poissonbootstrap1000_noSyst.pkl'
    },
    'Pythia' : {
        'proj' : '/ceph/submit/data/user/s/srothman/EEC/Oct14_2024/Pythia/EECproj/hists_file0to410_poissonbootstrap1000_noSyst.pkl',
        'res3' : '/ceph/submit/data/user/s/srothman/EEC/Oct14_2024/Pythia/EECres3/hists_file0to410_poissonbootstrap1000_noSyst.pkl',
        'res4tee' : '/ceph/submit/data/user/s/srothman/EEC/Oct14_2024/Pythia/EECres4tee/hists_file0to410_poissonbootstrap1000_noSyst.pkl',
        'res4dipole' : '/ceph/submit/data/user/s/srothman/EEC/Oct14_2024/Pythia/EECres4dipole/hists_file0to410_poissonbootstrap1000_noSyst.pkl'
    },
    'Pythia_nospin' : {
        'proj' : '/ceph/submit/data/user/s/srothman/EEC/Oct14_2024/Pythia_nospin/EECproj/hists_file0to291_poissonbootstrap1000_noSyst.pkl',
        'res3' : '/ceph/submit/data/user/s/srothman/EEC/Oct14_2024/Pythia_nospin/EECres3/hists_file0to291_poissonbootstrap1000_noSyst.pkl',
        'res4tee' : '/ceph/submit/data/user/s/srothman/EEC/Oct14_2024/Pythia_nospin/EECres4tee/hists_file0to291_poissonbootstrap1000_noSyst.pkl',
        'res4dipol' : '/ceph/submit/data/user/s/srothman/EEC/Oct14_2024/Pythia_nospin/EECres4dipole/hists_file0to291_poissonbootstrap1000_noSyst.pkl'
    },
    'Pythia_nobdecay' : {
        'proj' : '/ceph/submit/data/user/s/srothman/EEC/Oct14_2024/Pythia_nobdecay/EECproj/hists_file0to252_poissonbootstrap1000_noSyst.pkl',
        'res3' : '/ceph/submit/data/user/s/srothman/EEC/Oct14_2024/Pythia_nobdecay/EECres3/hists_file0to252_poissonbootstrap1000_noSyst.pkl',
        'res4tee': '/ceph/submit/data/user/s/srothman/EEC/Oct14_2024/Pythia_nobdecay/EECres4tee/hists_file0to252_poissonbootstrap1000_noSyst.pkl',
        'res4dipole' : '/ceph/submit/data/user/s/srothman/EEC/Oct14_2024/Pythia_nobdecay/EECres4dipole/hists_file0to252_poissonbootstrap1000_noSyst.pkl'
    }
}

##res4tee plots
#with open(paths['Herwig']['res4tee'], 'rb') as f:
#    res4tee_Herwig = pickle.load(f)['nominal']['reco']
#
#with open(paths['Herwig_nospin']['res4tee'], 'rb') as f:
#    res4tee_Herwig_nospin = pickle.load(f)['nominal']['reco']
#
#with open(paths['Herwig_nobdecay']['res4tee'], 'rb') as f:
#    res4tee_Herwig_nobdecay = pickle.load(f)['nominal']['reco']
#
#with open(paths['Pythia']['res4tee'], 'rb') as f:
#    res4tee_Pythia = pickle.load(f)['nominal']['reco']
#
#with open(paths['Pythia_nospin']['res4tee'], 'rb') as f:
#    res4tee_Pythia_nospin = pickle.load(f)['nominal']['reco']
#
#with open(paths['Pythia_nobdecay']['res4tee'], 'rb') as f:
#    res4tee_Pythia_nobdecay = pickle.load(f)['nominal']['reco']
#
#print("HERWIG")
#res4.plot_res4(res4tee_Herwig, 0, 2, 3)
#
#print("HERWIG_NOSPIN")
#res4.plot_res4(res4tee_Herwig_nospin, 0, 2, 3)
#
#asdfklj

#print("HERWIG_NOBDECAY")
#res4.plot_res4(res4tee_Herwig_nobdecay, 0, 1, 3)
#res4.plot_res4(res4tee_Herwig_nobdecay, 1, 1, 3)
#
#print("PYTHIA")
#res4.plot_res4(res4tee_Pythia, 0, 1, 3)
#res4.plot_res4(res4tee_Pythia, 1, 1, 3)
#
#print("PYTHIA_NOSPIN")
#res4.plot_res4(res4tee_Pythia_nospin, 0, 1, 3)
#res4.plot_res4(res4tee_Pythia_nospin, 1, 1, 3)
#
#print("PYTHIA_NOBDECAY")
#res4.plot_res4(res4tee_Pythia_nobdecay, 0, 1, 3)
#res4.plot_res4(res4tee_Pythia_nobdecay, 1, 1, 3)
#
#asdfkljafsd
#
#res4.plot_res4(res4tee_Herwig, 0, 1, 1)
#res4.plot_res4(res4tee_Herwig, 0, 1, 2)
#res4.plot_res4(res4tee_Herwig, 0, 1, 3)
#res4.plot_res4(res4tee_Herwig, 0, 1, 4)
#res4.plot_res4(res4tee_Herwig, 0, 1, 5)
#res4.plot_res4(res4tee_Herwig, 0, 1, 6)
#res4.plot_res4(res4tee_Herwig, 0, 1, 7)


#proj plots
with open(paths['Herwig']['proj'], 'rb') as f:
    proj_Herwig = pickle.load(f)['nominal']['reco']

with open(paths['Herwig_nospin']['proj'], 'rb') as f:
    proj_Herwig_nospin = pickle.load(f)['nominal']['reco']

with open(paths['Herwig_nobdecay']['proj'], 'rb') as f:
    proj_Herwig_nobdecay = pickle.load(f)['nominal']['reco']

with open(paths['Pythia']['proj'], 'rb') as f:
    proj_Pythia = pickle.load(f)['nominal']['reco']

with open(paths['Pythia_nospin']['proj'], 'rb') as f:
    proj_Pythia_nospin = pickle.load(f)['nominal']['reco']

with open(paths['Pythia_nobdecay']['proj'], 'rb') as f:
    proj_Pythia_nobdecay = pickle.load(f)['nominal']['reco']

proj.compareProjectedEEC(
    [proj_Pythia, proj_Pythia],
    [None]*2,
    [{'order' : 0, 'pt' : 1, 'btag' : 0}, {'order' : 0, 'pt' : 1, 'btag' : 1}],
    density=True,
    label_l = ['Pythia udsgc', 'Pythia b'],
    color_l = ['blue', 'red'],
    isData=False,
)
proj.compareProjectedEEC(
    [proj_Pythia_nobdecay, proj_Pythia_nobdecay],
    [None]*2,
    [{'order' : 0, 'pt' : 1, 'btag' : 0}, {'order' : 0, 'pt' : 1, 'btag' : 1}],
    density=True,
    label_l = ['Pythia_nobdecay udsgc', 'Pythia_nobdecay b'],
    color_l = ['blue', 'red'],
    isData=False,
)
proj.compareProjectedEEC(
    [proj_Herwig, proj_Herwig],
    [None]*2,
    [{'order' : 0, 'pt' : 1, 'btag' : 0}, {'order' : 0, 'pt' : 1, 'btag' : 1}],
    density=True,
    label_l = ['Herwig udsgc', 'Herwig b'],
    color_l = ['blue', 'red'],
    isData=False,
)
proj.compareProjectedEEC(
    [proj_Herwig_nobdecay, proj_Herwig_nobdecay],
    [None]*2,
    [{'order' : 0, 'pt' : 1, 'btag' : 0}, {'order' : 0, 'pt' : 1, 'btag' : 1}],
    density=True,
    label_l = ['Herwig_nobdecay udsgc', 'Herwig_nobdecay b'],
    color_l = ['blue', 'red'],
    isData=False,
)
proj.compareProjectedEEC(
        [proj_Herwig, proj_Herwig_nospin, proj_Herwig_nobdecay],
        [None]*3,
        [{'order' : 0, 'pt' : 1, 'btag' : 0}] * 3,
        density=True,
        label_l = ['Herwig', 'Herwig_nospin', 'Herwig_nobdecay'],
        color_l = ['blue', 'green', 'red'],
        isData=False,
)
proj.compareProjectedEEC(
        [proj_Pythia, proj_Pythia_nospin, proj_Pythia_nobdecay],
        [None]*3,
        [{'order' : 0, 'pt' : 1, 'btag' : 0}] * 3,
        density=True,
        label_l = ['Pythia', 'Pythia_nospin', 'Pythia_nobdecay'],
        color_l = ['blue', 'green', 'red'],
        isData=False,
)
proj.compareProjectedEEC(
        [proj_Herwig, proj_Herwig_nospin, proj_Herwig_nobdecay],
        [None]*3,
        [{'order' : 0, 'pt' : 1, 'btag' : 1}] * 3,
        density=True,
        label_l = ['Herwig', 'Herwig_nospin', 'Herwig_nobdecay'],
        color_l = ['blue', 'green', 'red'],
        isData=False,
)
proj.compareProjectedEEC(
        [proj_Pythia, proj_Pythia_nospin, proj_Pythia_nobdecay],
        [None]*3,
        [{'order' : 0, 'pt' : 1, 'btag' : 1}] * 3,
        density=True,
        label_l = ['Pythia', 'Pythia_nospin', 'Pythia_nobdecay'],
        color_l = ['blue', 'green', 'red'],
        isData=False,
)

