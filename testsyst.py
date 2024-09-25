import numpy as np
import proj
import matplotlib.pyplot as plt
from util import setup_plain_canvas

#reco_nom = np.load('../EECunfold/output/nominal/reco.npy')
#covreco_nom = np.load('../EECunfold/output/nominal/covreco.npy')
#
#forward_nom = np.load('../EECunfold/output/nominal/forward.npy')
#covforward_nom = np.load('../EECunfold/output/nominal/covforward.npy')
#
#reco_up = np.load('../EECunfold/output/idsfUp/reco.npy')
#covreco_up = np.load('../EECunfold/output/idsfUp/covreco.npy')
#
#forward_up = np.load('../EECunfold/output/idsfUp/forward.npy')
#covforward_up = np.load('../EECunfold/output/idsfUp/covforward.npy')
#
#reco_down = np.load('../EECunfold/output/idsfDown/reco.npy')
#covreco_down = np.load('../EECunfold/output/idsfDown/covreco.npy')
#
#forward_down = np.load('../EECunfold/output/idsfDown/forward.npy')
#covforward_down = np.load('../EECunfold/output/idsfDown/covforward.npy')

syst_impacts = []
syst_impact_uncs = []
names = []
for syst in ['wt_idsf', 'wt_isosf', 'wt_triggersf', 'wt_prefire']:
    syst_impacts.append(np.load('/data/submit/srothman/EECunfold/output/%s_impact.npy' % syst))
    syst_impact_uncs.append(np.load('/data/submit/srothman/EECunfold/output/%s_impact_unc.npy' % syst))
    names.append(syst[3:])

colors = ['red', 'blue', 'green', 'purple']

fig, ax = setup_plain_canvas(False)
for i in range(len(syst_impacts)):
    proj.plotProjectedEEC(
        syst_impacts[i],
        syst_impact_uncs[i],
        binning = {
            'order' : 0,
            'pt' : 3
        },
        density = True,
        label = names[i],
        color = colors[i],
        impact=True,
        ax = ax
    )
plt.legend(loc='upper left')
plt.tight_layout()
plt.savefig('syst_impacts.pdf')
plt.show()

