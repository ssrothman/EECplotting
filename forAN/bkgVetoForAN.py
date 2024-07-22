from util import *
from kin import *

# efficiency table
def bkgVetoEfficiency(samplename, maxNumLooseB, maxMET, Blevel):
    H = get_hist(samplename, 'Kinematics', 'noBkgVeto', 'nominal')['selvar']

    if Blevel == 'loose':
        H = H.project('numLooseB', 'MET')
    elif Blevel == 'medium':
        H = H.project('numMediumB', 'MET')
    elif Blevel == 'tight':
        H = H.project('numTightB', 'MET')
    else:
        raise ValueError("Invalid Btagging level")

    endMETbin = H.axes['MET'].index(maxMET)
    endBbin = H.axes['numLooseB'].index(maxNumLooseB)
    
    N = H.values(flow=True).sum()
    Npass = H[:endBbin:sum, :endMETbin:sum].value

    return Npass/N

with open("bkgVetoForAN/table.tex", 'w') as f:
    f.write("\\begin{tabular}{l|l}\n")
    f.write("Sample & Veto efficiency [\\%]\\\\\n")
    f.write("\\hline\n")
    for sample in reversed(order_by_jet_yield(MCsamplenames, 'noBkgVeto')):
        f.write("%s & %0.1f\\\\\n"%(get_label(sample), bkgVetoEfficiency(sample, 2, 50, 'loose')*100))
    f.write("\\end{tabular}\n")

# MET plots
MCstackdriver(MCsamplenames, 
              ['nominal']*len(samples),
              'noBkgVeto',
              'selvar', 'MET',
              density=False,
              plotsteps=True,
              normPerBin=False,
              folder='bkgVetoForAN')

MCstackdriver(MCsamplenames, 
              ['nominal']*len(samples),
              'noBkgVeto',
              'selvar', 'MET',
              density=False,
              plotsteps=True,
              normPerBin=True,
              folder='bkgVetoForAN')

# numLooseB plots
MCstackdriver(MCsamplenames, 
              ['nominal']*len(samples),
              'noBkgVeto',
              'selvar', 'numLooseB',
              density=False,
              plotsteps=True,
              normPerBin=False,
              folder='bkgVetoForAN')

MCstackdriver(MCsamplenames, 
              ['nominal']*len(samples),
              'noBkgVeto',
              'selvar', 'numLooseB',
              density=False,
              plotsteps=True,
              normPerBin=True,
              folder='bkgVetoForAN')

