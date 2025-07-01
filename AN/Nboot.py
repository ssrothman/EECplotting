import pickle

basepath = '/ceph/submit/data/group/cms/store/user/srothman/EEC/Apr_23_2025/Pythia_HTsum/EECres4tee/CONSTRUCTED_RECO/Apr_23_2025_Pythia_HTsum_REPLACEHERE_nominal_nominal'

Hpath = '/ceph/submit/data/group/cms/store/user/srothman/EEC/Apr_23_2025/Pythia_HTsum/EECres4tee/hists_0to0/nominal/reco_nominal_nominal_boot11675_HTSUM.pkl'

with open(Hpath, 'rb') as f:
    H = pickle.load(f)

import plotters.res4
import os
os.makedirs('AN/Nboot', exist_ok=True)

plotters.res4.covmat_trace(H, step=50, savefig='AN/Nboot/Nboot')
plotters.res4.squared_error_vs_Nboot(basepath, 500, 11500, 500, savefig='AN/Nboot/Nboot')
plotters.res4.eigenvals_vs_Nboot(basepath, 500, 11500, 500, savefig='AN/Nboot/Nboot')
