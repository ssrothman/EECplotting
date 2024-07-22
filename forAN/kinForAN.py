from kin import dataMCdriver, dataMCdriver_varySyst
from util import *

thesampls = MCsamplenames[:-1] +  ['DYJetsToLL_allHT']

testH = get_hist("DYJetsToLL_HT-2500toInf", "Kinematics", 
                 'scanSyst', 'nominal')

for Hname in testH:
    if Hname in ['config', 'sumwt', 'numjet', 'sumwt_pass']:
        continue

    for AXname in testH[Hname].axes.name:
        for key in systematics:
            dataMCdriver_varySyst(thesampls, 
                                  systematics[key]['systs'],
                                  'scanSyst',
                                  systematics[key]['colors'],
                                  Hname, AXname,
                                  folder='kinForAN')

            dataMCdriver(thesampls, 'nominal',
                         'scanSyst',
                         Hname, AXname,
                         plotsteps=True,
                         folder='kinForAN')
#
#dataMCdriver(thesampls, 'nominal',
#             'scanSyst',
#             'Z', 'y',
#             plotsteps=True,
#             folder='ZkinForAN')
#
#dataMCdriver(thesampls, 'nominal',
#             'scanSyst',
#             'Z', 'pt',
#             plotsteps=True,
#             folder='ZkinForAN')
