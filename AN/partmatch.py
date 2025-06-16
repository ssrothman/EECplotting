import plotters.kin

from importlib import reload
import numpy as np
import json

skimmer = 'Kinematics'
which = 'part'

plotter = plotters.kin.KinPlotManager()

plotter.toggle_show(False)

datasets_MC = plotters.kin.setup_datasets_MC(skimmer, which)

plotter.add_MC(datasets_MC['HT'])

plotter.setup_savefig("AN/partmatch")

lowEtaCut = plotters.kin.LessThanCut(
    plotters.kin.UFuncVariable("eta", np.abs),
    1.0
)
highEtaCut = plotters.kin.GreaterThanCut(
    plotters.kin.UFuncVariable("eta", np.abs),
    1.0
)
matchedCut = plotters.kin.EqualsCut(
    plotters.kin.UFuncVariable("nMatches", lambda x : x >= 1),
    1
)

lowPtBin = plotters.kin.TwoSidedCut(
    'pt', 
    0,
    1
)
medPtBin = plotters.kin.TwoSidedCut(
    'pt', 
    1,
    10
)
highPtBin = plotters.kin.GreaterThanCut(
    'pt', 
    10,
)

for ptbin in [lowPtBin, medPtBin, highPtBin]:
    for etabin in [lowEtaCut, highEtaCut]:
        plotter.plot_variable(
            plotters.kin.ResolutionVariable(
                'matchPhi',
                'phi'
            ),
            cut = plotters.kin.AndCuts(
                matchedCut,
                etabin,
                ptbin
            ),
            noResolved=True,
            cut_text=True,
            force_xlim = [-0.02, 0.02]
        )
        plotter.plot_variable(
            plotters.kin.ResolutionVariable(
                'matchEta',
                'eta'
            ),
            cut = plotters.kin.AndCuts(
                matchedCut,
                etabin,
                ptbin
            ),
            noResolved=True,
            cut_text=True,
            force_xlim = [-0.02, 0.02]
        )
        plotter.plot_variable(
            plotters.kin.RelativeResolutionVariable(
                'matchPt',
                'pt'
            ),
            cut = plotters.kin.AndCuts(
                matchedCut,
                etabin,
                ptbin
            ),
            noResolved=True,
            cut_text=True,
            force_xlim = [-0.15, 0.15]
        )

for etabin in [lowEtaCut, highEtaCut]:
    plotter.plot_variable(
        plotters.kin.RateVariable(
            plotters.kin.UFuncVariable("nMatches", lambda x : x >= 1), 
            "pt"
        ),
        cut = etabin,
        noResolved=True,
        cut_text=True
    )
