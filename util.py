import json
import os
import os.path
import numpy as np
import matplotlib.pyplot as plt
import mplhep as hep

plt.style.use(hep.style.CMS)

APPROVAL_TEXT = "Work in progress"

with open("config.json", 'r') as f:
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

def is_integer(axname):
    return config['isinteger'][axname]

def ax_in_all(binning_l, axname):
    for binning in binning_l:
        if axname not in binning:
            return False
    return True

def ax_same_in_all(binning_l, axname):
    for binning in binning_l:
        if axname not in binning:
            return False
        if binning[axname] != binning_l[0][axname]:
            return False
    return True

def ax_extent(axname):
    ans = len(config['binedges'][axname]) - 1
    if has_overflow(axname):
        ans += 1
    if has_underflow(axname):
        ans += 1
    return ans

def binning_AND(binning_l):
    ans = {}
    for axname in config['projshape']:
        if ax_same_in_all(binning_l, axname):
            ans[axname] = binning_l[0][axname]

    return ans

def binning_name(binning):
    ans = ''
    for axname in binning.keys():
        if type(binning[axname]) is int:
            ans += '%s%d' % (axname, binning[axname])
        elif type(binning[axname]) is slice:
            ans += '%s%dto%d' % (axname, binning[axname].start, binning[axname].stop)
        else:
            raise ValueError("Unknown binning type")
        ans += '_'
    if ans[-1] == '_':
        ans = ans[:-1]

    return ans

def binning_string(binning):
    ans = ''
    for axname in binning.keys():
        if type(binning[axname]) is int:
            theslice = slice(binning[axname], binning[axname]+1)
        else:
            theslice = binning[axname]
        
        edges = get_ax_edges(axname)
        if has_underflow(axname):
            edges = np.concatenate(([-np.inf], edges))
        if has_overflow(axname):
            edges = np.concatenate((edges, [np.inf]))

        centers = (edges[1:] + edges[:-1])/2

        if is_integer(axname):
            ans += '%s$ = %d$' % (get_ax_label(axname), centers[theslice])
        else:
            ans += '$%g < $%s$ < %g$' % (edges[theslice.start], 
                                               get_ax_label(axname), 
                                               edges[theslice.stop])

        ans += '\n'

    if ans[-1] == '\n':
        ans = ans[:-1]

    return ans

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
            if type(binning[axname]) is int:
                slice_list += [slice(binning[axname], binning[axname]+1)]
            else:
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
            if type(binning1[axname]) is int:
                slice_list += [slice(binning1[axname], binning1[axname]+1)]
            else:
                slice_list += [binning1[axname]]
        else:
            slice_list += [slice(None)]

        if axname != onto:
            sum_list += [i]

    for i, axname in enumerate(config['projshape']):
        if axname in binning2:
            if type(binning2[axname]) is int:
                slice_list += [slice(binning2[axname], binning2[axname]+1)]
            else:
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
            if type(binningReco[axname]) is int:
                slice_list += [slice(binningReco[axname], binningReco[axname]+1)]
            else:
                slice_list += [binningReco[axname]]
        else:
            slice_list += [slice(None)]

    for i,axname in enumerate(config['projshape']):
        if axname in binningGen:
            if type(binningGen[axname]) is int:
                slice_list += [slice(binningGen[axname], binningGen[axname]+1)]
            else:
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
            if type(binning[axname]) is int:
                slice_list += [slice(binning[axname], binning[axname]+1)]
            else:
                slice_list += [binning[axname]]
        else:
            slice_list += [slice(None)]

    slice_tuple = tuple(slice_list*2)

    size = np.prod(cov.shape[:len(config['projshape'])])
    return cov[slice_tuple].reshape((size, size))

def savefig(outname):
    print("Saving to", outname)
    os.makedirs(os.path.dirname(outname), exist_ok=True)
    plt.tight_layout()
    plt.savefig(outname, dpi=300, format='png', bbox_inches='tight')
    plt.close()
