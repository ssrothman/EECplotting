import datasets
import numpy as np
from tqdm import tqdm

reco = datasets.get_pickled_histogram("Apr_23_2025", 'Pythia_inclusive', "EECres4tee", "nominal", 'nominal', 'reco', shuffle_boots=True).values(flow=True)

nom = reco[0].ravel()
boots = reco[1:]
Nboot = boots.shape[0]
boots = boots.reshape((Nboot, -1))

nomsum = nom.sum()
bootsum = boots.sum(axis=1, keepdims=True)
boots = boots * nomsum / bootsum

DY = boots - nom[None, :]

step = 10
traces = []
Nbs = []
for Nb in tqdm(range(step, Nboot, 10)):
    cov = DY[:Nb].T @ DY[:Nb] / Nb
    trace = np.trace(cov)
    traces.append(trace)
    Nbs.append(Nb)

import matplotlib.pyplot as plt
import json
import mplhep as hep
with open("config/config.json", 'r') as f:
    config = json.load(f)

plt.style.use(hep.style.CMS)

fig = plt.figure(figsize=config['Figure_Size'])
hep.cms.label(data=False, label=config['Approval_Text'])

plt.errorbar(Nbs, traces, fmt='o')
plt.xlabel("Number of Bootstrap Samples")
plt.ylabel("Trace of Covariance Matrix")
plt.savefig("AN/Nboot/trace_cov.png", format='png', bbox_inches='tight', dpi=300)
plt.show()
