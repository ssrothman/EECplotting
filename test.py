from proj import plotProjectedEEC, compareProjectedEEC, plotProjectedCorrelation, plotProjectedTransfer
from util import setup_plain_canvas
import numpy as np
import matplotlib.pyplot as plt

def closure():
    pythia_reco = np.load('../EECunfold/testpythianpy/reco.npy')
    pythia_covreco = np.load('../EECunfold/testpythianpy/covreco.npy')
    pythia_gen = np.load('../EECunfold/testpythianpy/gen.npy')
    pythia_covgen = np.load('../EECunfold/testpythianpy/covgen.npy')

    herwig_reco = np.load('../EECunfold/testherwignpy/reco.npy')
    herwig_covreco = np.load('../EECunfold/testherwignpy/covreco.npy')
    herwig_gen = np.load('../EECunfold/testherwignpy/gen.npy')
    herwig_covgen = np.load('../EECunfold/testherwignpy/covgen.npy')

    pp_unfolded = np.load('../EECunfold/pythiapythia/unfolded.npy')
    pp_covunfolded = np.load('../EECunfold/pythiapythia/covunfolded.npy')
    pp_forward = np.load('../EECunfold/pythiapythia/forwarded.npy')
    pp_covforward = np.load('../EECunfold/pythiapythia/covforwarded.npy')

    ph_unfolded = np.load('../EECunfold/pythiaherwig/unfolded.npy')
    ph_covunfolded = np.load('../EECunfold/pythiaherwig/covunfolded.npy')
    ph_forward = np.load('../EECunfold/pythiaherwig/forwarded.npy')
    ph_covforward = np.load('../EECunfold/pythiaherwig/covforwarded.npy')

    for ptbin in range(6):
        compareProjectedEEC(pythia_reco, pythia_covreco,
                            pp_forward, pp_covforward,
                            binning1={'order' : slice(0,1),
                                      'pt' : slice(ptbin,ptbin+1)},
                            binning2={'order' : slice(0,1),
                                      'pt' : slice(ptbin,ptbin+1)},
                            label1='pythia reco',
                            label2='pythia gen * pythia transfer',
                            wrt='dR')

        compareProjectedEEC(pythia_gen, pythia_covgen,
                            pp_unfolded, pp_covunfolded,
                            binning1={'order' : slice(0,1),
                                      'pt' : slice(ptbin,ptbin+1)},
                            binning2={'order' : slice(0,1),
                                      'pt' : slice(ptbin,ptbin+1)},
                            label1='pythia gen',
                            label2='pythia reco * (pythia transfer)^-1',
                            wrt='dR')

        compareProjectedEEC(herwig_reco, herwig_covreco,
                            ph_forward, ph_covforward,
                            binning1={'order' : slice(0,1),
                                      'pt' : slice(ptbin,ptbin+1)},
                            binning2={'order' : slice(0,1),
                                      'pt' : slice(ptbin,ptbin+1)},
                            label1='herwig reco',
                            label2='herwig gen * pythia transfer',
                            wrt='dR')

        compareProjectedEEC(herwig_gen, herwig_covgen,
                            ph_unfolded, ph_covunfolded,
                            binning1={'order' : slice(0,1),
                                      'pt' : slice(ptbin,ptbin+1)},
                            binning2={'order' : slice(0,1),
                                      'pt' : slice(ptbin,ptbin+1)},
                            label1='herwig gen',
                            label2='herwig reco * (pythia transfer)^-1',
                            wrt='dR')

def TEST_TRANSFERPROJ():
    unfolded = np.load('test/unfolded.npy')
    covunfolded = np.load('test/covunfolded.npy')

    gen = np.load('test/genpure.npy')
    covgen = np.load('test/covgen.npy')

    plotProjectedCorrelation(covunfolded,
                             binning1={},
                             binning2={},
                             wrt='dR')

    plotProjectedCorrelation(covunfolded,
                             binning1={},
                             binning2={},
                             wrt='order')

    plotProjectedCorrelation(covunfolded,
                             binning1={},
                             binning2={},
                             wrt='btag')

    plotProjectedCorrelation(covunfolded,
                             binning1={},
                             binning2={},
                             wrt='pt')

    compareProjectedEEC(unfolded, covunfolded,
                        gen, covgen,
                        binning1={},
                        binning2={},
                        label1='unfolded',
                        label2='gen',
                        wrt='dR')

    compareProjectedEEC(unfolded, covunfolded,
                        gen, covgen,
                        binning1={'order' : slice(0,1),
                                  'pt' : slice(0,1)},
                        binning2={'order' : slice(0,1),
                                  'pt' : slice(0,1)},
                        label1='unfolded',
                        label2='gen',
                        wrt='dR')

    compareProjectedEEC(unfolded, covunfolded,
                        gen, covgen,
                        binning1={'order' : slice(0,1),
                                  'pt' : slice(1,2)},
                        binning2={'order' : slice(0,1),
                                  'pt' : slice(1,2)},
                        label1='unfolded',
                        label2='gen',
                        wrt='dR')

    compareProjectedEEC(unfolded, covunfolded,
                        gen, covgen,
                        binning1={'order' : slice(0,1),
                                  'pt' : slice(2,3)},
                        binning2={'order' : slice(0,1),
                                  'pt' : slice(2,3)},
                        label1='unfolded',
                        label2='gen',
                        wrt='dR')

    compareProjectedEEC(unfolded, covunfolded,
                        gen, covgen,
                        binning1={'order' : slice(0,1),
                                  'pt' : slice(3,4)},
                        binning2={'order' : slice(0,1),
                                  'pt' : slice(3,4)},
                        label1='unfolded',
                        label2='gen',
                        wrt='dR')

    forward = np.load('test/forwarded.npy')
    covforward = np.load('test/covforwarded.npy')

    reco = np.load('test/recopure.npy')
    covreco = np.load('test/covreco.npy')

    plotProjectedCorrelation(covforward,
                             binning1={},
                             binning2={},
                             wrt='dR')

    plotProjectedCorrelation(covforward,
                             binning1={},
                             binning2={},
                             wrt='order')

    plotProjectedCorrelation(covforward,
                             binning1={},
                             binning2={},
                             wrt='btag')

    plotProjectedCorrelation(covforward,
                             binning1={},
                             binning2={},
                             wrt='pt')

    compareProjectedEEC(forward, covforward,
                        reco, covreco,
                        binning1={},
                        binning2={},
                        label1='forward',
                        label2='reco',
                        wrt='dR')

    compareProjectedEEC(forward, covforward,
                        reco, covreco,
                        binning1={'order' : slice(0,1),
                                  'pt' : slice(0,1)},
                        binning2={'order' : slice(0,1),
                                  'pt' : slice(0,1)},
                        label1='forward',
                        label2='reco',
                        wrt='dR')

    compareProjectedEEC(forward, covforward,
                        reco, covreco,
                        binning1={'order' : slice(0,1),
                                  'pt' : slice(1,2)},
                        binning2={'order' : slice(0,1),
                                  'pt' : slice(1,2)},
                        label1='forward',
                        label2='reco',
                        wrt='dR')

    compareProjectedEEC(forward, covforward,
                        reco, covreco,
                        binning1={'order' : slice(0,1),
                                  'pt' : slice(2,3)},
                        binning2={'order' : slice(0,1),
                                  'pt' : slice(2,3)},
                        label1='forward',
                        label2='reco',
                        wrt='dR')

    compareProjectedEEC(forward, covforward,
                        reco, covreco,
                        binning1={'order' : slice(0,1),
                                  'pt' : slice(3,4)},
                        binning2={'order' : slice(0,1),
                                  'pt' : slice(3,4)},
                        label1='forward',
                        label2='reco',
                        wrt='dR')

    transfer = np.load('test/transfer_over_gen.npy')
    plotProjectedTransfer(transfer,
                          binningGen={},
                          binningReco={})

    plotProjectedTransfer(transfer,
                          binningGen={'order' : slice(0,1)},
                          binningReco={'order' : slice(0,1)})

    plotProjectedTransfer(transfer,
                          binningGen={'order' : slice(0,1),
                                      'btag' : slice(0,1)},
                          binningReco={'order' : slice(0,1),
                                       'btag' : slice(0,1)})

    plotProjectedTransfer(transfer,
                          binningGen={'order' : slice(0,1),
                                      'btag' : slice(0,1),
                                      'pt' : slice(2,3)},
                          binningReco={'order' : slice(0,1),
                                       'btag' : slice(0,1),
                                       'pt' : slice(2,3)})

def TEST_CORRELPROJ():
    covx1 = np.load('test/covreco.npy')
    plotProjectedCorrelation(covx1, 
                             binning1={'order' : slice(0,1)},
                             binning2=None,
                             wrt='dR')

    plotProjectedCorrelation(covx1, 
                             binning1={},
                             binning2=None,
                             wrt='order')

    plotProjectedCorrelation(covx1, 
                             binning1={},
                             binning2=None,
                             wrt='pt')

    plotProjectedCorrelation(covx1, 
                             binning1={},
                             binning2=None,
                             wrt='btag')

    plotProjectedCorrelation(covx1, 
                             binning1={'order' : slice(0,1)},
                             binning2={'order' : slice(1,2)},
                             wrt='dR')

def TEST_COMPAREPROJ():
    x1 = np.load('test/reco.npy')
    covx1 = np.load('test/covreco.npy')
    x2 = np.load('test/gen.npy')
    covx2 = np.load('test/covgen.npy')

    compareProjectedEEC(x1, covx1,
                        x2, covx2,
                        binning1={'order' : slice(0,1)},
                        binning2={'order' : slice(0,1)},
                        label1='reco',
                        label2='gen',
                        wrt='dR')
    compareProjectedEEC(x1, covx1,
                        x2, covx2,
                        binning1={'order' : slice(0,1)},
                        binning2={'order' : slice(0,1)},
                        label1='reco',
                        label2='gen',
                        wrt='pt')
    
    compareProjectedEEC(x1, covx1,
                        x1, covx1,
                        binning1={'order' : slice(0,1)},
                        binning2={'order' : slice(0,1)},
                        label1='reco',
                        label2='gen',
                        wrt='dR')
    compareProjectedEEC(x1, covx1,
                        x1, covx1,
                        binning1={'order' : slice(1,2)},
                        binning2={'order' : slice(0,1)},
                        label1='reco',
                        label2='gen',
                        wrt='dR')

def TEST_PLOTPROJ():
    x = np.load('test/reco.npy')
    covx = np.load('test/covreco.npy')

    fig, ax = setup_plain_canvas(False)

    plotProjectedEEC(x, covx,
                     wrt='dR', 
                     binning={'order' : slice(0,1)},
                     ax=ax,
                     label='Second order')
    plotProjectedEEC(x, covx, 
                     wrt='dR', 
                     binning={'order' : slice(1,2)},
                     ax=ax,
                     label='Third order')
    plotProjectedEEC(x, covx, 
                     wrt='dR', 
                     binning={'order' : slice(2,3)},
                     ax=ax,
                     label='Fourth order')
    plotProjectedEEC(x, covx, 
                     wrt='dR', 
                     binning={'order' : slice(3,4)},
                     ax=ax,
                     label='Fifth other')
    plotProjectedEEC(x, covx, 
                     wrt='dR', 
                     binning={'order' : slice(4,5)},
                     ax=ax,
                     label='Sixth order')
    plt.show()

    plotProjectedEEC(x, covx,
                     wrt='order', 
                     logwidth=False)
    plt.show()

    plotProjectedEEC(x, covx, 
                     wrt='btag', 
                     logwidth=False)
    plt.show()

    plotProjectedEEC(x, covx,
                     wrt='pt', 
                     logwidth=False)
    plt.show()

#TEST_TRANSFERPROJ()
#TEST_CORRELPROJ()
#TEST_COMPAREPROJ()
#TEST_PLOTPROJ()
closure()
