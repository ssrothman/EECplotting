import matplotlib.pyplot as plt
import numpy as np
import os.path
import mplhep as hep
from matplotlib.colors import LogNorm, Normalize
import correctionlib.schemav2 as cs

from util import setup_ratiopad_canvas, setup_plain_canvas, get_ax_edges, has_overflow, has_underflow, should_logx, should_logy, setup_cbar_canvas, get_ax_label, get_hist, get_color, get_label, get_lumi, get_xsec, order_by_jet_yield, savefig

def makeBeffJson(samplenames, folder):
    beffs = {}
    for level in ['loose', 'medium' ,'tight']:
        xedges, eff, vareff = getBeff(samplenames, level)
        eff_up = eff + np.sqrt(vareff)
        eff_down = eff - np.sqrt(vareff)

        eff_up = np.clip(eff_up, 0, 1)
        eff_down = np.clip(eff_down, 0, 1)

        beffs[level] = {
            'nominal': eff,
            'up' : eff_up,
            'down': eff_down
        }

    ptedges = xedges.tolist()
    ptedges[0] = 0
    ptedges[-1] = 'inf'

    testH = get_hist(samplenames[0], 'Beff', 'noBSF', 'nominal')['Beff']
    etaedges = testH.axes['eta'].edges.tolist()

    flavindexing = {
        0 : 0,
        4 : 1,
        5 : 2
    }
    
    levelnodes = []
    for level in ['loose', 'medium', 'tight']:
        systnodes = []
        for syst in ['nominal', 'up', 'down']:
            flavnodes = []
            for flav in [0, 4, 5]:
                flavnodes.append(cs.CategoryItem(
                    key = flav,
                    value = cs.MultiBinning(
                        nodetype='multibinning',
                        inputs=['pt', 'abseta'],
                        edges=[ptedges, etaedges],
                        content=beffs[level][syst][flavindexing[flav], :, :].ravel().tolist(),
                        flow='error'
                    )
                ))
            flavdata = cs.Category(
                nodetype='category',
                input='hadronFlavour',
                content=flavnodes
            )

            systnodes.append(cs.CategoryItem(
                key=syst,
                value=flavdata
            ))

        systdata = cs.Category(
            nodetype='category',
            input='syst',
            content=systnodes
        )
        
        levelnodes.append(cs.CategoryItem(
            key=level,
            value=systdata
        ))
    leveldata = cs.Category(
        nodetype='category',
        input='level',
        content=levelnodes
    )

    corr = cs.Correction(
        name='Beff',
        version=1,
        description = 'B tagger efficiency for AK4 CHS jets',
        inputs=[
            cs.Variable(name='level', type='string', 
                        description='Btagging level'),
            cs.Variable(name='syst', type='string', 
                        description='One of nominal, up, down'),
            cs.Variable(name='pt', type='real',
                        description='CHS jet pT [GeV]'),
            cs.Variable(name='abseta', type='real',
                        description='CHS jet |eta|'),
            cs.Variable(name='hadronFlavour', type='int',
                        description='CHS jet hadron flavour'+
                                    ' (0: udsg, 4: c, 5: b)')
        ],
        output=cs.Variable(name='efficiency', type='real', 
                           description='B tagger pass rate'),
        data = leveldata
    )

    import rich
    rich.print(corr)

    cset = cs.CorrectionSet(
        schema_version = 2,
        corrections = [corr]
    )

    fname = os.path.join(folder, 'Beff.json')
    with open(fname, 'w') as f:
        f.write(cset.model_dump_json(exclude_unset=True, indent=4))

def getBeff(samplenames, level):
    H = None
    for samplename in samplenames:
        #note that we don't care about getting the overall pT spectrum right
        #we just care about the pass probability GIVEN a jet in a pT bin
        #so in order to get the counting statistics right
        #we don't weight by xsec
        Hnext = get_hist(samplename, 'Beff', 'noBSF', 'nominal')['Beff']

        if H is None:
            H = Hnext
        else:
            H += Hnext

    if level == 'loose': 
        H = H.project("flavor", "pt", 'eta', 'looseB')
    elif level == 'medium':
        H = H.project("flavor", "pt", 'eta', 'mediumB')
    elif level == 'tight':
        H = H.project("flavor", "pt", 'eta', 'tightB')

    vals = H.values(flow=True)[:-1,1:,1:-1, 1:-1]

    k = vals[:,:,:,1]
    N = vals.sum(axis=3)
    eff = k/N
    vareff = ((k+1)*(k+2))/((N+2)*(N+3))-np.square(k+1)/np.square(N+2)

    xedges = H.axes['pt'].edges
    xedges = np.concatenate([xedges, [1e4]])

    return xedges, eff, np.sqrt(vareff)

def plotBeff(samplenames, level, eta, folder=None):
    xedges, eff, vareff = getBeff(samplenames, level)

    fig, ax = setup_plain_canvas(False)

    ax.set_xlabel(get_ax_label('pt', 'Beff'))
    ax.set_ylabel("Efficiency")

    xcenters = (xedges[1:] + xedges[:-1])/2
    xerr = (xedges[1:] - xedges[:-1])/2

    for i, flavor in enumerate(['usdg', 'c', 'b']):
        ax.errorbar(xcenters, eff[i,:,eta], 
                    xerr=xerr, yerr=vareff[i,:,eta], 
                    fmt='o', label=flavor)

    labeltext = ''
    if eta == 0:
        labeltext = 'Barrel'
    else:
        labeltext = 'Endcaps'

    if level == 'loose':
        labeltext += ';\tLoose WP'
    elif level == 'medium':
        labeltext += ';\tMedium WP'
    elif level == 'tight':
        labeltext += ';\tTight WP'

    ax.text(0.05, 0.95, labeltext,
            transform=ax.transAxes, 
            ha='left', va='center')

    ax.set_xscale('log')
    ax.set_ylim(0, 1.1)
    ax.axhline(1, color='black', linestyle='--')

    ax.legend(loc='best', 
              edgecolor='black',
              facecolor='white',
              frameon=True)

    if folder is None:
        plt.show()
    else:
        fname = 'Beff_%s_eta%d.png'%(level, eta)
        savefig(os.path.join(folder, fname))
