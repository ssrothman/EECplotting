import plotters.res4
import pickle
import os

def validate(path, Hgen, Hreco):
    with open(os.path.join(path, 'minimization_result', 'Hunf_boot5000.pkl'), 'rb') as f:
        Hunf = pickle.load(f)

    with open(os.path.join(path, 'minimization_result', 'Hfwd_boot5000.pkl'), 'rb') as f:
        Hfwd = pickle.load(f)

    #print("chi^2 reco:", plotters.res4.chi2_2d(Hreco, Hfwd))
    print("chi^2 gen:", plotters.res4.chi2_2d(Hgen, Hunf))

    for ptbin in range(5):
        plotters.res4.compare_1d(
                Hunf, Hgen,
                "Unfolded", "Gen",
                2, ptbin, 
                r=0, c=0, 
                shiftx=0.1
        )

       # plotters.res4.compare_1d(
       #         Hfwd, Hreco,
       #         "Forward", "Reco",
       #         2, ptbin, 
       #         r=0, c=0, 
       #         shiftx=0.1
       # )
