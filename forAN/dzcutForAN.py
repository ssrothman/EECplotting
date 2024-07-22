from proj import *
from util import get_hist

wdz = get_hist("DYJetsToLL", "EECproj", 'testdz', 'nominal')
nodz = get_hist("DYJetsToLL_nodz", "EECproj", 'testdz', 'nominal')

x_l = [wdz['recopure'], nodz['recopure']]
covx_l = [wdz['covreco'], nodz['covreco']]
binning_l = [
    {
        'order' : 0,
        'pt' : 3,
        'btag' : 0
    }
]*2
label_l = ['with PV cut', 'without PV cut']
color_l = ['blue', 'red']

compareProjectedEEC(
   x_l, covx_l,
   binning_l,
   True,
   label_l,
   color_l,
   False,
   'dR',
   True,
   folder='dzcutForAN'
)

binning_l[0]['btag'] = 1
binning_l[1]['btag'] = 1
compareProjectedEEC(
   x_l, covx_l,
   binning_l,
   True,
   label_l,
   color_l,
   False,
   'dR',
   True,
   folder='dzcutForAN'
)

