import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize
import pickle
import numpy as np
import awkward as ak

with open("/ceph/submit/data/user/s/srothman/EEC/Oct14_2024/Pythia/EECres4tee/hists_file0to410_poissonbootstrap1000_noSyst.pkl", 'rb') as f:
    pythia = pickle.load(f)['nominal']['reco']

with open("/ceph/submit/data/user/s/srothman/EEC/Oct14_2024/Pythia_nospin/EECres4tee/hists_file0to291_poissonbootstrap1000_noSyst.pkl", 'rb') as f:
    pythia_nospin = pickle.load(f)['nominal']['reco']

with open("/ceph/submit/data/user/s/srothman/EEC/Oct14_2024/Herwig/EECres4tee/hists_file0to241_poissonbootstrap1000_noSyst.pkl", 'rb') as f:
    herwig = pickle.load(f)['nominal']['reco']

with open("/ceph/submit/data/user/s/srothman/EEC/Oct14_2024/Herwig_nospin/EECres4tee/hists_file0to198_poissonbootstrap1000_noSyst.pkl", 'rb') as f:
    herwig_nospin = pickle.load(f)['nominal']['reco']


def get_val_err(data):
    #use the poisson replicas
    nominal = data[:,0]
    replicas = data[:,1:]
    residuals = replicas - nominal[:,None]
    #calculate the mean and standard deviation of the replicas
    mean = np.mean(residuals, axis=1)
    std = np.std(residuals, axis=1)

    return nominal, std

def plot_res4_angular(data, btag, ptbin, RLbin, rs, 
                      ax=None, label=None, color=None):
    val, err = get_val_err(data)

    theval = val[0, btag, ptbin, RLbin, 1:-1, 1:-1]
    theerr = err[0, btag, ptbin, RLbin, 1:-1, 1:-1]

    nr, ntheta = theval.shape

    r_edges = np.linspace(0, 1, nr+1)
    theta_edges = np.linspace(0, np.pi/2, ntheta+1)

    r_mid = (r_edges[1:] + r_edges[:-1])/2
    theta_mid = (theta_edges[1:] + theta_edges[:-1])/2

    area = 0.5*(r_edges[1:,None]**2 - r_edges[:-1,None]**2)*(theta_edges[None,1:] - theta_edges[None,:-1])

    theval = theval/area
    theerr = theerr/area

    mean = np.mean(theval)
    theval = theval/mean
    theerr = theerr/mean

    angular_sum = np.sum(theval, axis=1)
    angular_sum_err = np.sqrt(np.sum(theerr**2, axis=1))

    angular_val = theval/angular_sum[:,None]
    angular_err = theerr/angular_sum[:,None]

    theta_to_plot = np.concatenate([theta_mid,
                                    theta_mid+np.pi/2,
                                    theta_mid+np.pi,
                                    theta_mid+3*np.pi/2])
    val_to_plot = np.concatenate([angular_val[rs],
                                  angular_val[rs][::-1],
                                  angular_val[rs],
                                  angular_val[rs][::-1]])
    err_to_plot = np.concatenate([angular_err[rs],
                                  angular_err[rs][::-1],
                                  angular_err[rs],
                                  angular_err[rs][::-1]])

    ax.errorbar(theta_to_plot, val_to_plot, 
                yerr=err_to_plot, 
                fmt='o', label=label, color=color) 
    ax.axvline(np.pi/2, color='black', linestyle='--')
    ax.axvline(np.pi, color='black', linestyle='--')
    ax.axvline(3*np.pi/2, color='black', linestyle='--')
    ax.set_xlabel('$\\theta$')
    ax.set_ylabel('Angular profile')

    def cos2x(x, A, B):
        return A+B*np.cos(2*x)

    import scipy.optimize as opt
    popt, pcov = opt.curve_fit(cos2x, theta_mid, angular_val[rs], sigma=angular_err[rs])
    plt.plot(theta_to_plot, cos2x(theta_to_plot, *popt), color=color)

def compare_res4_angular(data_l, btag_l, ptbin_l, RLbin_l, rs, labels):
    colors = [(0.12156862745098039, 0.4666666666666667, 0.7058823529411765), 
              (1.0, 0.4980392156862745, 0.054901960784313725)]
    fig, ax = plt.subplots()
    for data, btag, ptbin, RLbin, label, color in zip(data_l, btag_l, ptbin_l, RLbin_l, labels, colors):
        plot_res4_angular(data, btag, ptbin, RLbin, rs, ax=ax, label=label, color=color)

    #legend inside of opaque box
    ax.legend(loc='upper right', frameon=True)

    plt.show()

def plot_res4(data, btag, ptbin, RLbin, 
              vmin=None, vmax=None,
              normscheme='log',
              cmap='viridis'):
    val, err = get_val_err(data)

    theval = val[0, btag, ptbin, RLbin, 1:-1, 1:-1]
    theerr = err[0, btag, ptbin, RLbin, 1:-1, 1:-1]


    nr, ntheta = theval.shape

    r_edges = np.linspace(0, 1, nr+1)
    theta_edges = np.linspace(0, np.pi/2, ntheta+1)

    r_mid = (r_edges[1:] + r_edges[:-1])/2
    theta_mid = (theta_edges[1:] + theta_edges[:-1])/2

    area = 0.5*(r_edges[1:,None]**2 - r_edges[:-1,None]**2)*(theta_edges[None,1:] - theta_edges[None,:-1])

    theval = theval/area
    theerr = theerr/area

    mean = np.mean(theval)
    theval = theval/mean
    theerr = theerr/mean



   # fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})

   # if normscheme == 'log':
   #     norm = LogNorm(vmin=vmin, vmax=vmax)
   # elif normscheme == 'linear':
   #     norm = Normalize(vmin=vmin, vmax=vmax)

   # pc1 = ax.pcolormesh(theta_edges, r_edges,
   #                     theval,
   #                     cmap='viridis',
   #                     shading='auto',
   #                     norm=norm)
   # pc2 = ax.pcolormesh(np.pi-theta_edges, r_edges,
   #                     theval,
   #                     cmap='viridis',
   #                     shading='auto',
   #                     norm=norm)
   # pc3 = ax.pcolormesh(np.pi+theta_edges, r_edges,
   #                     theval,
   #                     cmap='viridis',
   #                     shading='auto',
   #                     norm=norm)
   # pc4 = ax.pcolormesh(2*np.pi-theta_edges, r_edges,
   #                     theval,
   #                     cmap='viridis',
   #                     shading='auto',
   #                     norm=norm)
   # plt.colorbar(pc1, ax=ax)
   # plt.show()

    angular_sum = np.sum(theval, axis=1)
    angular_sum_err = np.sqrt(np.sum(theerr**2, axis=1))

    angular_val = theval/angular_sum[:,None]
    angular_err = theerr/angular_sum[:,None]

    #fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})

    #norm = Normalize(vmin=None, vmax=None)
    #pc1 = ax.pcolormesh(theta_edges, r_edges,
    #                    angular_val,
    #                    cmap='viridis',
    #                    shading='auto',
    #                    norm=norm)
    #pc2 = ax.pcolormesh(np.pi-theta_edges, r_edges,
    #                    angular_val,
    #                    cmap='viridis',
    #                    shading='auto',
    #                    norm=norm)
    #pc3 = ax.pcolormesh(np.pi+theta_edges, r_edges,
    #                    angular_val,
    #                    cmap='viridis',
    #                    shading='auto',
    #                    norm=norm)
    #pc4 = ax.pcolormesh(2*np.pi-theta_edges, r_edges,
    #                    angular_val,
    #                    cmap='viridis',
    #                    shading='auto',
    #                    norm=norm)
    #plt.colorbar(pc1, ax=ax)
    #plt.show()

    fig, (ax0, ax1) = plt.subplots(2,1)

    ax0.errorbar(r_mid, angular_sum, yerr=angular_sum_err, fmt='o')
    ax0.set_xlabel('r')
    ax0.set_ylabel('Radial profile')
    ax0.set_yscale('log')
    
    for r in [4]:
        x_to_plot = np.concatenate([theta_mid, 
                                    theta_mid+np.pi/2, 
                                    theta_mid+np.pi,
                                    theta_mid+3*np.pi/2])
        y_to_plot = np.concatenate([angular_val[r],
                                    angular_val[r][::-1], 
                                    angular_val[r], 
                                    angular_val[r][::-1]])
        err_to_plot = np.concatenate([angular_err[r],
                                      angular_err[r][::-1],
                                      angular_err[r],
                                      angular_err[r][::-1]])
        ax1.errorbar(x_to_plot, y_to_plot, 
                     yerr=err_to_plot,
                     fmt='o', label=f'r={r_mid[r]:.2f}')
    ax1.set_xlabel(r'$\theta$')
    ax1.set_ylabel('Angular profile')
    ax1.legend()
    
    plt.show()
