from util import *
import os.path

def plot_pulls(path_to_matrices,
               folder=None, fprefix=None):
    syst_names = config['systs']

    g_pred = np.load(os.path.join(path_to_matrices,'g_pred.npy'))
    cov_g_pred = np.load(os.path.join(path_to_matrices,'cov_g_pred.npy'))

    Nsyst = len(syst_names)
    pulls = g_pred[-Nsyst:]
    dpulls = np.sqrt(np.diag(cov_g_pred)[-Nsyst:])

    fig, ax = setup_plain_canvas(False)

    ax.errorbar(pulls, range(Nsyst), xerr=dpulls, fmt='o', color='black', markersize=10)
    ax.axvline(0, color='black', linestyle='--')
    ax.set_yticks(range(Nsyst))
    ax.set_yticklabels(syst_names, fontsize=26, rotation=15)
    ax.set_xlabel('Pull', fontsize=26)

    maxpull = np.max(np.abs(pulls))
    if maxpull < 2:
        ax.set_xlim(-2, 2)
    else:
        ax.set_xlim(-maxpull-0.5, maxpull+0.5)

    ax.fill_betweenx([-0.5, Nsyst-0.5], -1, 1, color='gray', alpha=0.5)

    plt.tight_layout()

    if folder is not None:
        fname = folder
        if fprefix is not None:
            fname = os.path.join(folder, "pulls_%s.png" % fprefix)
        else:
            fname = os.path.join(folder, "pulls.png")
        savefig(fname)
    else:
        plt.show()


plot_pulls("/data/submit/srothman/EECunfold/unfolded_triggersfUp",   
           "pullplots", "triggersfUp")
plot_pulls("/data/submit/srothman/EECunfold/unfolded_triggersfDown", 
           "pullplots", "triggersfDown")
plot_pulls("/data/submit/srothman/EECunfold/unfolded_prefireUp",     
           "pullplots", "prefireUp")
plot_pulls("/data/submit/srothman/EECunfold/unfolded_prefireDown",   
           "pullplots", "prefireDown")
plot_pulls("/data/submit/srothman/EECunfold/unfolded_isosfUp",       
           "pullplots", "isosfUp")
plot_pulls("/data/submit/srothman/EECunfold/unfolded_isosfDown",     
           "pullplots", "isosfDown")
plot_pulls("/data/submit/srothman/EECunfold/unfolded_idsfUp",        
           "pullplots", "idsfUp")
plot_pulls("/data/submit/srothman/EECunfold/unfolded_idsfDown",      
           "pullplots", "idsfDown")
