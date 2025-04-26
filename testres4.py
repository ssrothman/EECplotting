from datasets import get_pickled_histogram, get_unfolded_histogram
import plotters.res4
from importlib import reload
import numpy as np
import matplotlib.pyplot as plt

#which = 'reco'
#Hpythia = get_pickled_histogram('Apr_01_2025', 'Pythia_inclusive', 'EECres4tee', 'nominal', 'nominal', which)
#Hherwig = get_pickled_histogram('Apr_01_2025', 'Herwig_inclusive', 'EECres4tee', 'nominal', 'nominal', which)
#HdataA = get_pickled_histogram('Apr_01_2025', 'DATA_2018A', 'EECres4tee', 'nominal', 'nominal', which)
#HdataB = get_pickled_histogram('Apr_01_2025', 'DATA_2018B', 'EECres4tee', 'nominal', 'nominal', which)
#HdataC = get_pickled_histogram('Apr_01_2025', 'DATA_2018C', 'EECres4tee', 'nominal', 'nominal', which)
#HdataD = get_pickled_histogram('Apr_01_2025', 'DATA_2018D', 'EECres4tee', 'nominal', 'nominal', which)
#Hdata = HdataA + HdataB + HdataC + HdataD

Hpythia_gen = get_pickled_histogram('Apr_01_2025', 'Pythia_inclusive', 'EECres4tee', 'nominal', 'nominal', 'gen')
Hpythia_unmatchedGen = get_pickled_histogram('Apr_01_2025', 'Pythia_inclusive', 'EECres4tee', 'nominal', 'nominal', 'unmatchedGen')
Hpythia_untransferedGen = get_pickled_histogram('Apr_01_2025', 'Pythia_inclusive', 'EECres4tee', 'nominal', 'nominal', 'untransferedGen')
Hpythia_pureGen = Hpythia_gen - Hpythia_unmatchedGen - Hpythia_untransferedGen

Hpythia_reco = get_pickled_histogram('Apr_01_2025', 'Pythia_inclusive', 'EECres4tee', 'nominal', 'nominal', 'reco')
Hpythia_unmatchedReco = get_pickled_histogram('Apr_01_2025', 'Pythia_inclusive', 'EECres4tee', 'nominal', 'nominal', 'unmatchedReco')
Hpythia_untransferedReco = get_pickled_histogram('Apr_01_2025', 'Pythia_inclusive', 'EECres4tee', 'nominal', 'nominal', 'untransferedReco')
Hpythia_pureReco = Hpythia_reco - Hpythia_unmatchedReco - Hpythia_untransferedReco

Hpythia_transfer = get_pickled_histogram('Apr_01_2025', 'Pythia_inclusive', 'EECres4tee', 'nominal', 'nominal', 'transfer')

#Hherwig_gen = get_pickled_histogram('Apr_01_2025', 'Herwig_inclusive', 'EECres4tee', 'nominal', 'nominal', 'gen')
#Hherwig_unmatchedGen = get_pickled_histogram('Apr_01_2025', 'Herwig_inclusive', 'EECres4tee', 'nominal', 'nominal', 'unmatchedGen')
#Hherwig_untransferedGen = get_pickled_histogram('Apr_01_2025', 'Herwig_inclusive', 'EECres4tee', 'nominal', 'nominal', 'untransferedGen')
#Hherwig_pureGen = Hherwig_gen - Hherwig_unmatchedGen - Hherwig_untransferedGen

unf = get_unfolded_histogram("pythia_unf_pythia_unfolded")
fwd = get_unfolded_histogram("pythia_unf_pythia_forward")
tra = get_unfolded_histogram("pythia_unf_pythia_transfer")

