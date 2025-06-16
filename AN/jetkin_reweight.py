import plotters.kin
import numpy as np

skimmer = 'Kinematics'
which = 'jet'

datasets_data = plotters.kin.setup_datasets_DATA(skimmer, which)
datasets_MC = plotters.kin.setup_datasets_MC(skimmer, which)

MC = datasets_MC['allMC']
data = datasets_data['allDATA']

MC.set_samplewt_MC(data.lumi)

var = plotters.kin.Variable('pt')
cut = plotters.kin.NoCut()
weighting = plotters.kin.Variable('evtwt_nominal')

MC.evaluate_table(var, cut, weighting)
data.evaluate_table(var, cut, weighting)

global_min = 30
global_max = 1500
bins=20

MC.fill_hist(global_min, global_max, bins, True)
data.fill_hist(global_min, global_max, bins, True)

from util import histogram_ratio
ratio, ratioerr = histogram_ratio(MC.H, data.H)
ratio = ratio[1:-1]
ratioerr = ratioerr[1:-1]

edges = MC.H.axes[0].edges
logedges = np.log10(edges)
logcenters =0.5*(logedges[1:] + logedges[:-1])
logwidths = (logedges[1:] - logedges[:-1])/2

import ratiofit
res = ratiofit.minimize(logcenters, ratio, ratioerr, ratiofit.poly4, 4)

import matplotlib.pyplot as plt
invratio, invratioerr = histogram_ratio(data.H, MC.H)
invratio = invratio[1:-1]
invratioerr = invratioerr[1:-1]
plt.errorbar(logcenters, invratio, yerr=invratioerr, xerr=logwidths, fmt='o', label="data/Pythia")
finex = np.linspace(np.min(logedges), np.max(logedges), 100)
plt.plot(finex, ratiofit.invpoly4(finex, res.x), label='Fit')
plt.legend(loc='best')
plt.xlabel("$\\log_{10} p_T^{Jet}$")
plt.ylabel("Data/MC scale factor")
import mplhep as hep
import json
with open("config/config.json", 'r') as f:
    config = json.load(f)
hep.cms.label(data=True, label=config['Approval_Text'], year=config['Year'], lumi='%0.2f'%data.lumi)
plt.savefig("AN/jetkin_reweight/pythia.png", format='png', bbox_inches='tight', dpi=300)
plt.clf()

MC_herwig = datasets_MC['allMCHerwig']
MC_herwig.set_samplewt_MC(data.lumi)

MC_herwig.evaluate_table(var, cut, weighting)
MC_herwig.fill_hist(global_min, global_max, bins, True)

ratioH, ratioerrH = histogram_ratio(MC_herwig.H, data.H)
ratioH = ratioH[1:-1]
ratioerrH = ratioerrH[1:-1]

resH = ratiofit.minimize(logcenters, ratioH, ratioerrH, ratiofit.poly5, 5)
invratioH, invratioerrH = histogram_ratio(data.H, MC_herwig.H)
invratioH = invratioH[1:-1]
invratioerrH = invratioerrH[1:-1]
plt.errorbar(logcenters, invratioH, yerr=invratioerrH, xerr=logwidths, fmt='o', label="data/Herwig")
plt.plot(finex, ratiofit.invpoly5(finex, resH.x), label='Fit')
plt.legend(loc='best')
plt.xlabel("$\\log_{10} p_T^{Jet}$")
plt.ylabel("Data/MC scale factor")
hep.cms.label(data=True, label=config['Approval_Text'], year=config['Year'], lumi='%0.2f'%data.lumi)
plt.savefig("AN/jetkin_reweight/herwig.png", format='png', bbox_inches='tight',dpi=300)
plt.clf()

pythia_jetpt_SF = '''
import numpy as np

class THESF:
    @staticmethod
    def evaluate(pt, res):
        x = np.log10(pt)
        p = res.x
        poly = p[0] + p[1]*x + p[2]*x*x + p[3]*x*x*x 
        return 1/poly
'''

herwig_jetpt_SF = '''
import numpy as np

class THESF:
    @staticmethod
    def evaluate(pt, res):
        x = np.log10(pt)
        p = res.x
        poly = p[0] + p[1]*x + p[2]*x*x + p[3]*x*x*x + p[4]*x*x*x*x + p[5]*x*x*x*x*x
        return 1/poly
'''

import dill as pickle
with open("kinSF/pythia_jetpt_SF.py", 'w') as f:
    f.write(pythia_jetpt_SF)

with open('kinSF/herwig_jetpt_SF.py', 'w') as f:
    f.write(herwig_jetpt_SF)

with open("kinSF/fit_pythia.pkl", 'wb') as f:
    pickle.dump(res, f)

with open("kinSF/fit_herwig.pkl", 'wb') as f:
    pickle.dump(resH, f)
