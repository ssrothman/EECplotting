

import simon_mpl_util as smu

pythia4 = smu.ParquetDataset('/ceph/submit/data/group/cms/store/user/srothman/EEC/Apr_23_2025/Pythia_inclusive/Kinematics/hists_file0to198_MC/nominal/event/')
pythia4.set_label("Pythia inclusive")
pythia4.set_color("red")
pythia4.set_xsec(6077.22)
import pickle
with open("/ceph/submit/data/group/cms/store/user/srothman/EEC/Apr_23_2025/Pythia_inclusive/Count/hists_file0to396_MC_nominal.pickle", 'rb') as f:
    pythia4.override_num_events(pickle.load(f)['num_evt'])

pythia = smu.ParquetDataset('/ceph/submit/data/group/cms/store/user/srothman/EEC/Apr_23_2025/Pythia_inclusive/Kinematics/hists_file0to198_MC/nominal/event/')
pythia.set_label("Pythia inclusive x1/3")
pythia.set_color("blue")
pythia.set_xsec(6077.22/3)
import pickle
with open("/ceph/submit/data/group/cms/store/user/srothman/EEC/Apr_23_2025/Pythia_inclusive/Count/hists_file0to396_MC_nominal.pickle", 'rb') as f:
    pythia.override_num_events(pickle.load(f)['num_evt'])

pythia2 = smu.ParquetDataset('/ceph/submit/data/group/cms/store/user/srothman/EEC/Apr_23_2025/Pythia_inclusive/Kinematics/hists_file0to198_MC/nominal/event/')
pythia2.set_label("Pythia inclusive x2/3")
pythia2.set_color("orange")
pythia2.set_xsec(6077.22*2/3)
import pickle
with open("/ceph/submit/data/group/cms/store/user/srothman/EEC/Apr_23_2025/Pythia_inclusive/Count/hists_file0to396_MC_nominal.pickle", 'rb') as f:
    pythia2.override_num_events(pickle.load(f)['num_evt'])

pythia3 = smu.DatasetStack([pythia, pythia2])
pythia3.set_label("[Stack] Pythia inclusive")
pythia3.set_color("green")

data = smu.ParquetDataset('/ceph/submit/data/group/cms/store/user/srothman/EEC/Apr_23_2025/DATA_2018D/Kinematics/hists_file0to768_data/nominal/event/')
data.set_label("Data")
data.set_color("black")
data.set_lumi(31.833472186)

variable = smu.BasicVariable('Zpt')
cut = smu.NoCut()
wtMC = smu.BasicVariable("evtwt_nominal")
wtDT = smu.ConstantVariable(1.0)
wtMC_alt = smu.BasicVariable("evtwt_FSRUp")

smu.plot_histogram(
    variable, cut,
    [wtMC, wtMC_alt],
    [pythia3,pythia3],
    smu.AutoBinning(),
    ['Nominal', 'FSR Up'],
    density=False,
    output_path = 'test',
    logy=True,
    logx=True
)