import plotters.kin

from importlib import reload
import numpy as np
import json

skimmer = 'Kinematics'
which = 'jet'

plotter = plotters.kin.KinPlotManager()

plotter.toggle_show(False)

datasets_MC = plotters.kin.setup_datasets_MC(skimmer, which)

plotter.add_MC(datasets_MC['HT'])

plotter.setup_savefig("AN/jetmatch")

plotter.plot_variable(
    plotters.kin.RateVariable("matched", "pt"),
    noResolved=True,
)
plotter.plot_variable(
    plotters.kin.RelativeResolutionVariable(
        'matchPt',
        'pt'
    ),
    cut = plotters.kin.AndCuts(
        plotters.kin.EqualsCut(
            'matched',
            1
        ),
        plotters.kin.TwoSidedCut(
            'pt',
            500,
            1500
        )
    ),
    noResolved = True,
    force_xlim = np.asarray([-1, 1]),
    cut_text = True,
)
plotter.plot_variable(
    plotters.kin.RelativeResolutionVariable(
        'matchPt',
        'pt'
    ),
    cut = plotters.kin.AndCuts(
        plotters.kin.EqualsCut(
            'matched',
            1
        ),
        plotters.kin.TwoSidedCut(
            'pt',
            50,
            100
        )
    ),
    noResolved = True,
    force_xlim = np.asarray([-1, 1]),
    cut_text = True,
)
plotter.plot_variable(
    plotters.kin.RelativeResolutionVariable(
        'matchPt',
        'pt'
    ),
    cut = plotters.kin.AndCuts(
        plotters.kin.EqualsCut(
            'matched',
            1
        ),
        plotters.kin.TwoSidedCut(
            'pt',
            100,
            500
        )
    ),
    noResolved = True,
    force_xlim = np.asarray([-1, 1]),
    cut_text = True,
)
