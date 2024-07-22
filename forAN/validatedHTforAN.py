from kin import *
from util import *

samples = ['DYJetsToLL_HT-0to70',
           'DYJetsToLL_HT-70to100',
           'DYJetsToLL_HT-100to200',
           'DYJetsToLL_HT-200to400',
           'DYJetsToLL_HT-400to600',
           'DYJetsToLL_HT-600to800',
           'DYJetsToLL_HT-800to1200',
           'DYJetsToLL_HT-1200to2500',
           'DYJetsToLL_HT-2500toInf',]

from itertools import cycle
cycol = cycle('bgrcmk')

HTstack = []
xsecs = []

for sample in samples:
    H = get_hist(sample, 'Kinematics', 'scanSyst', 'nominal')
    HTstack.append(H)
    xsecs.append(get_xsec(sample))

Hdict_l_l = [
        [get_hist('DYJetsToLL', 'Kinematics', 'scanSyst', 'nominal')],
        HTstack, 
        #[get_hist('DYJetsToLL_allHT', 'Kinematics', 'scanSyst', 'nominal')],
]
xsecs_l_l = [
    [get_xsec('DYJetsToLL')],
    xsecs, 
]
color_l = [
        'r',
        'g', 
        #'b'
]
label_l = ['DYJetsToLL',
           'DYJetsToLL_HT', 
           #'DYJetsToLL_allHT'
]

#for i in range(3):
#    print(label_l[i], xsecs_l_l[i])

compareKin(Hdict_l_l, xsecs_l_l, "Z", 'mass', color_l, label_l, density=False)

plt.show()
