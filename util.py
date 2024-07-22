import json
import pickle
import os
import os.path
import numpy as np
import matplotlib.pyplot as plt
import mplhep as hep
from iadd import iadd
from mult import Hdict_mult

plt.style.use(hep.style.CMS)

APPROVAL_TEXT = "Work in progress"

MCsamplenames = ['ST_tW_top', 'ST_tW_antitop', 'ST_t_top', 'ST_t_antitop', 
           'WZ', 'WW', 'ZZ',
           'TTTo2L2Nu', 
           'DYJetsToLL']
HTsamplenames = ['DYJetsToLL_HT-0to70',
                 'DYJetsToLL_HT-70to100',
                 'DYJetsToLL_HT-100to200',
                 'DYJetsToLL_HT-200to400',
                 'DYJetsToLL_HT-400to600',
                 'DYJetsToLL_HT-600to800',
                 'DYJetsToLL_HT-800to1200',
                 'DYJetsToLL_HT-1200to2500',
                 'DYJetsToLL_HT-2500toInf']
HTpBKGsamplenames = MCsamplenames[:-1] + HTsamplenames

systematics = {
    'theory' : {
        'systs' : ['nominal', 'wt_scaleUp', 'wt_scaleDown',
                              'wt_PDFaSDown', 'wt_PDFaSUp'],
        'colors' : ['black', 'blue', 'red',
                    'green', 'purple']
    },
    'muons' : {
        'systs' : ['nominal', 'wt_idsfUp', 'wt_idsfDown', 
                              'wt_isosfUp', 'wt_isosfDown',
                              'wt_triggersfUp', 'wt_triggersfDown',
                              'wt_prefireUp', 'wt_prefireDown'],
        'colors' : ['black', 'blue', 'red', 
                             'green', 'purple',
                             'orange', 'cyan',
                             'magenta', 'yellow']
    },
    "PS" : {
        "systs" : ['nominal', 'wt_ISRUp', 'wt_ISRDown',
                   'wt_FSRUp', 'wt_FSRDown'],
        "colors" : ['black', 'blue', 'red', 
                    'green', 'purple']
    },
    'btag' : {
        "systs" : ['nominal', 'wt_btagSF_effUp', 'wt_btagSF_effDown',
                   'wt_btagSF_sfUp', 'wt_btagSF_sfDown'],
        "colors" : ['black', 'blue', 'red',
                    'green', 'purple']
    },
    'PU' : {
        "systs" : ['nominal', 'wt_PUUp', 'wt_PUDown'],
        "colors" : ['black', 'blue', 'red']
    },
    "jetmet" : {
        "systs" : ['nominal', 'JER_UP', 'JER_DN', 
                   'JES_UP', 'JES_DN', 
                   'UNCLUSTERED_UP', 'UNCLUSTERED_DN'],
        "colors" : ['black', 'blue', 'red',
                    'green', 'purple',
                    'orange', 'cyan']
    }
}

with open("config.json", 'r') as f:
    config = json.load(f)

with open("samples.json", 'r') as f:
    samples = json.load(f)

def list_systematics(samplename, which, tag):
    H = get_hist(samplename, which, tag, None)
    return list(H.keys())

def get_hist(samplename, which, tag, systematic='nominal'):
    if samplename == 'DATA':
        H = None
        for era in ["2018A", "2018B", '2018C', '2018D']:
            with open(samples[samplename]['hists'][which][tag][era], 'rb') as f:
                Hnext = pickle.load(f)
                if systematic is not None:
                    Hnext = Hnext[systematic]
            if H is None:
                H = Hnext
            else:
                H = iadd(H, Hnext)
        return H
    elif samplename == 'DYJetsToLL_allHT':
        H = None
        for HTsample in HTsamplenames:
            with open(samples[HTsample]['hists'][which][tag], 'rb') as f:
                Hnext = pickle.load(f)

            if systematic is not None:
                Hnext = Hnext[systematic]
                factor = get_xsec(HTsample)/Hnext['sumwt']
            else:
                factor = get_xsec(HTsample)/Hnext['nominal']['sumwt']

            Hnext = Hdict_mult(Hnext, factor)

            if H is None:
                H = Hnext
            else:
                H = iadd(H, Hnext)
        return H
    else:
        with open(samples[samplename]['hists'][which][tag], 'rb') as f:
            Hd = pickle.load(f)
        if systematic is None:
            return Hd
        elif systematic in Hd.keys():
            return Hd[systematic]
        else:
            print("WARNING: systematic '%s' not available in hist"%systematic)
            print("the offending hist is [%s][%s][%s]"%(samplename,which,tag))
            print("Defaulting to nominal")
            return Hd['nominal']


def get_xsec(sample):
    return samples[sample]['xsec']

def get_color(sample):
    return samples[sample]['color']

def get_label(sample):
    return samples[sample]['label']

def get_lumi():
    return samples['DATA']['lumi']

def get_jet_yield(samplename, histtag, systematic='nominal'):
    H = get_hist(samplename, 'Kinematics', histtag, systematic)
    xsec = get_xsec(samplename)
    factor = xsec/H['sumwt']
    return factor*H['numjet']

def order_by_jet_yield(samplelist, histtag):
    if type(samplelist) in [list, tuple]:
        return sorted(samplelist, key=lambda x: get_jet_yield(x, histtag))
    else:
        return zip(*sorted(samplelist, key=lambda x: get_jet_yield(x[0], histtag)))

def add_cms_label(ax, isData):
    if isData:
        hep.cms.label(ax=ax, data=True, label=APPROVAL_TEXT,
    year=2018, lumi=config['lumi'])
    else:
        hep.cms.label(ax=ax, data=False, label=APPROVAL_TEXT)


def setup_plain_canvas(isData):
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    add_cms_label(ax, isData)

    return fig, ax

def setup_ratiopad_canvas(isData):
    fig, (ax, rax) = plt.subplots(2, 1, figsize=(10, 10),
                                  sharex = True,
                                  gridspec_kw = {'height_ratios': [3, 1],
                                                 'hspace': 0.0})
    add_cms_label(ax, isData)

    return fig, (ax, rax)

def setup_cbar_canvas(isData):
    fig, (ax, cax) = plt.subplots(1, 2, figsize=(10.5, 10),
                                  gridspec_kw={'width_ratios': [10, 0.5],
                                               'wspace': 0.1})
    add_cms_label(ax, isData)
    return fig, (ax, cax)

def get_ax_label(axname, Hname):
    return config['axlabels'][axname][Hname]

def get_ax_edges(axname):
    return np.array(config['binedges'][axname]) 

def has_overflow(axname):
    return config['hasoverflow'][axname]

def has_underflow(axname):
    return config['hasunderflow'][axname]

def should_logx(axname):
    return config['should_logx'][axname]

def should_logy():
    return config['should_logy']

def is_integer(axname):
    return config['isinteger'][axname]

def savefig(outname):
    print("Saving to", outname)
    if os.path.dirname(outname) != '':
        os.makedirs(os.path.dirname(outname), exist_ok=True)
    plt.tight_layout()
    plt.savefig(outname, dpi=300, format='png', bbox_inches='tight')
    plt.close()
