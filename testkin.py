from datasets import get_dataset
import plotters.kin

which = 'reco'
df_pythia = get_dataset('Apr_01_2025', 'Pythia_inclusive', 'EECres4tee', 'nominal', which)
df_herwig = get_dataset('Apr_01_2025', 'Herwig_inclusive', 'EECres4tee', 'nominal', which)
#df_dataA = get_dataset('Apr_01_2025', 'DATA_2018A', 'EECres4tee', 'nominal', which)
#df_dataB = get_dataset('Apr_01_2025', 'DATA_2018B', 'EECres4tee', 'nominal', which)
#df_dataC = get_dataset('Apr_01_2025', 'DATA_2018C', 'EECres4tee', 'nominal', which)
df_dataD = get_dataset('Apr_01_2025', 'DATA_2018D', 'EECres4tee', 'nominal', which)

plotter = plotters.kin.KinPlotManager()
plotter.add_MC(df_pythia, 'Pythia')
plotter.add_MC(df_herwig, 'Herwig')
#plotter.add_data(df_dataA)
#plotter.add_data(df_dataB)
#plotter.add_data(df_dataC)
plotter.add_data(df_dataD)

plotter.plot_variable('pt')
