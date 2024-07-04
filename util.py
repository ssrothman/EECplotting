import json
import numpy as np
import matplotlib.pyplot as plt
import mplhep as hep

plt.style.use(hep.style.CMS)

APPROVAL_TEXT = "Work in progress"

with open("test/config.json", 'r') as f:
    config = json.load(f)

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

def get_ax_label(axname):
    return config['axlabels'][axname]

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

def project_projected(x, covx, binning, onto):
    '''
    x : np.array with the projected EEC
    binning: dict with binning to apply
        formatted like {'axname' : slice object}
    onto: target axis
    '''

    slice_list = []
    sum_list = []
    for i, axname in enumerate(config['projshape']):
        if axname in binning:
            slice_list += [binning[axname]]
        else:
            slice_list += [slice(None)]

        if axname != onto:
            sum_list += [i]

    slice_tuple = tuple(slice_list)
    slice_tuple_cov = tuple(slice_list*2)

    sum_list_cov = sum_list + [i + len(config['projshape']) for i in sum_list]

    projected_cov = np.sum(covx[slice_tuple_cov], 
                           axis=tuple(sum_list_cov))

    if x is not None:
        projected = np.sum(x[slice_tuple], 
                           axis=tuple(sum_list))
        return projected, projected_cov
    else:
        return projected_cov

def project_cov1x2_projected(covx, binning1, binning2, onto):
    slice_list = []
    sum_list = []

    for i, axname in enumerate(config['projshape']):
        if axname in binning1:
            slice_list += [binning1[axname]]
        else:
            slice_list += [slice(None)]

        if axname != onto:
            sum_list += [i]

    for i, axname in enumerate(config['projshape']):
        if axname in binning2:
            slice_list += [binning2[axname]]
        else:
            slice_list += [slice(None)]

        if axname != onto:
            sum_list += [i + len(config['projshape'])]

    slice_tuple = tuple(slice_list)
    return np.sum(covx[slice_tuple], 
                  axis=tuple(sum_list))

def bin_transfer_projected(transfer, binningGen, binningReco):
    slice_list = []

    for i,axname in enumerate(config['projshape']):
        if axname in binningReco:
            slice_list += [binningReco[axname]]
        else:
            slice_list += [slice(None)]

    for i,axname in enumerate(config['projshape']):
        if axname in binningGen:
            slice_list += [binningGen[axname]]
        else:
            slice_list += [slice(None)]

    slice_tuple = tuple(slice_list)
    sliced = transfer[slice_tuple]

    shapeReco = np.prod(sliced.shape[:len(config['projshape'])])
    shapeGen = np.prod(sliced.shape[len(config['projshape']):])

    return sliced.reshape((shapeReco, shapeGen))

def bin_cov_projected(cov, binning):
    slice_list = []

    for i, axname in enumerate(config['projshape']):
        if axname in binning:
            slice_list += [binning[axname]]
        else:
            slice_list += [slice(None)]

    slice_tuple = tuple(slice_list*2)

    size = np.prod(cov.shape[:len(config['projshape'])])
    return cov[slice_tuple].reshape((size, size))
