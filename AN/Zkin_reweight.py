import plotters.kin
import numpy as np

skimmer = 'Kinematics'
which = 'event'

datasets_data = plotters.kin.setup_datasets_DATA(skimmer, which)
datasets_MC = plotters.kin.setup_datasets_MC(skimmer, which)

MC = datasets_MC['allMC']
data = datasets_data['allDATA']

MC.set_samplewt_MC(data.lumi)

var1 = plotters.kin.Variable('Zpt')
var2 = plotters.kin.UFuncVariable('Zy', np.abs)
vars_l = [var1, var2]
cut = plotters.kin.NoCut()
weighting = plotters.kin.Variable('evtwt_nominal')

print("tables")
MC.evaluate_table_Nd(vars_l, cut, weighting)
data.evaluate_table_Nd(vars_l, cut, weighting)

global_min = [
    [0, 2, 5, 10, 20, 40, 50, 60, 70, 80, 90, 100, 
     120, 140, 160, 180, 200, 250, 300, 500, 750,
     2000, 10000, 100000],
    [0.0, 1.5, 2.0, 2.5]
]
global_max = [None, None]
nbins = [None, None]
logx = [None, None]

print("hists")
MC.fill_hist_Nd(global_min, global_max, nbins, logx)
data.fill_hist_Nd(global_min, global_max, nbins, logx)

print("ratio")
from util import histogram_ratio
import matplotlib.pyplot as plt
ratio, ratioerr = histogram_ratio(data.H, MC.H, flow=False)
ratio[~np.isfinite(ratio)] = 1.0

logptedges = np.log10(global_min[0])
logptcenters = 0.5 * (logptedges[1:] + logptedges[:-1])
logptwidths = (logptedges[1:] - logptedges[:-1]) / 2

yedges = global_min[1]

import mplhep as hep
plt.style.use(hep.style.CMS)

import json
with open("config/config.json", 'r') as f:
    config = json.load(f)

fig = plt.figure(figsize=config['Figure_Size'])

for i in range(len(global_min[1])-1):
    plt.errorbar(logptcenters, ratio[:,i], yerr=ratioerr[:,i], xerr=logptwidths, fmt='o', label=f'%g < Zy < %g'%(yedges[i], yedges[i+1]))

plt.xlabel("$\\log_{10} p_T^Z$")
plt.ylabel("Data/Pythia scale factor")
plt.ylim(0, 2)
plt.legend(loc='best')
hep.cms.label(data=True, label=config['Approval_Text'],
              year=config['Year'], lumi='%0.2f'%data.lumi)
plt.savefig("AN/Zkin_reweight/pythia_SF.png", format='png', bbox_inches='tight', dpi=300)
#plt.show()
plt.clf()

MCH = datasets_MC['allMCHerwig']
MCH.set_samplewt_MC(data.lumi)
MCH.evaluate_table_Nd(vars_l, cut, weighting)
MCH.fill_hist_Nd(global_min, global_max, nbins, logx)

ratio_herwig, ratioerr_herwig = histogram_ratio(data.H, MCH.H, flow=False)
ratio_herwig[~np.isfinite(ratio_herwig)] = 1.0
plt.figure(figsize=config['Figure_Size'])
for i in range(len(global_min[1])-1):
    plt.errorbar(logptcenters, ratio_herwig[:,i], yerr=ratioerr_herwig[:,i], xerr=logptwidths, fmt='o', label=f'%g < Zy < %g'%(yedges[i], yedges[i+1]))
plt.xlabel("$\\log_{10} p_T^Z$")
plt.ylabel("Data/Herwig scale factor")
plt.ylim(0, 2)
plt.legend(loc='best')
hep.cms.label(data=True, label=config['Approval_Text'],
              year=config['Year'], lumi='%0.2f'%data.lumi)
plt.savefig("AN/Zkin_reweight/herwig_SF.png", format='png', bbox_inches='tight', dpi=300)
#plt.show()
plt.clf()


import correctionlib.schemav2 as cs
corr_pythia = cs.Correction(
    name="Pythia_Zkinweight",
    version=1,
    inputs=[
        cs.Variable(name="log10_Zpt", type="real", description="log10 Z pt"),
        cs.Variable(name="Zy", type="real", description="Z rapidity"),
    ],
    output=cs.Variable(name="weight", type="real", description="Data/MC scale factor"),
    data = cs.MultiBinning(
        nodetype='multibinning',
        inputs=['log10_Zpt', 'Zy'],
        edges=[
            global_min[0],
            global_min[1]
        ],
        content = ratio.ravel().tolist(),
        flow='clamp'
    )
)
corr_herwig = cs.Correction(
    name="Herwig_Zkinweight",
    version=1,
    inputs=[
        cs.Variable(name="log10_Zpt", type="real", description="log10 Z pt"),
        cs.Variable(name="Zy", type="real", description="Z rapidity"),
    ],
    output=cs.Variable(name="weight", type="real", description="Data/MC scale factor"),
    data = cs.MultiBinning(
        nodetype='multibinning',
        inputs=['log10_Zpt', 'Zy'],
        edges=[
            global_min[0],
            global_min[1]
        ],
        content = ratio_herwig.ravel().tolist(),
        flow='clamp'
    )
)
cset = cs.CorrectionSet(
    schema_version=2,
    description="Zkin reweighting correction",
    corrections=[corr_pythia, 
                 corr_herwig]

)

with open("kinSF/Zkin.json", 'w') as f:
    f.write(cset.json(exclude_unset=True))
