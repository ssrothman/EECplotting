from util import *
from beff import plotBeff, makeBeffJson

HTsamplnames = ['DYJetsToLL_HT-0to70', 'DYJetsToLL_HT-70to100', 'DYJetsToLL_HT-100to200', 'DYJetsToLL_HT-200to400', 'DYJetsToLL_HT-400to600', 'DYJetsToLL_HT-600to800', 'DYJetsToLL_HT-800to1200', 'DYJetsToLL_HT-1200to2500', 'DYJetsToLL_HT-2500toInf']

#make binning table
testH = get_hist(HTsamplnames[0], 'Beff', 'noBSF', 'nominal')['Beff']
with open("beffForAN/binning.tex", 'w') as f:
    f.write('\\begin{tabular}{p{1.5cm}|p{6cm}}\n')
    f.write('Variable & Binning  \\\\\n')
    f.write('\\hline\n')
    f.write('$p_T$    & %s\\\\\n'%str(testH.axes['pt'].edges.tolist() + [np.inf]))
    f.write('$|\eta|$ & %s \\\\\n'%str(testH.axes['eta'].edges.tolist()))
    f.write('flavor   & [udsg, c, b]\n')
    f.write('\\end{tabular} \n')

makeBeffJson(HTsamplnames, 'beffForAN')
makeBeffJson(HTsamplnames, '/work/submit/srothman/EEC/postprocessing/corrections/Beff')

plotBeff(HTsamplnames, 'loose',  0, folder='beffForAN')
plotBeff(HTsamplnames, 'loose',  1, folder='beffForAN')
plotBeff(HTsamplnames, 'medium', 0, folder='beffForAN')
plotBeff(HTsamplnames, 'medium', 1, folder='beffForAN')
plotBeff(HTsamplnames, 'tight',  0, folder='beffForAN')
plotBeff(HTsamplnames, 'tight',  1, folder='beffForAN')
