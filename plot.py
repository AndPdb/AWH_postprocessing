import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import ticker, cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import copy

matplotlib.use('Agg')

plt.rcParams['xtick.labelsize']=20
plt.rcParams['ytick.labelsize']=20
plt.rcParams['figure.dpi'] = 300
colors = plt.rcParams["axes.prop_cycle"]()

WORKING_DIR       = "/nethome/apasqua/Workspace/AAC/5_MD_simulations/5_AWH_AF2/2D_contacts/custom_target/trial_2/"
plot_pullx        = False
plot_PMF          = True
awh_xvg          = "walker_1/awh_data/awh_t183300.xvg"

if plot_pullx:
    data = {}
    walkers = [ x for x in os.listdir(WORKING_DIR) if x.startswith('walker') ]
    #Check if pullx.npy present and up-to-date
    for walker in walkers:
        WALKER_DIR = os.path.join(WORKING_DIR,walker)
        pullxs = [ x for x in os.listdir(WALKER_DIR) if x.endswith('pullx.xvg') ]
        npy = os.path.join(WORKING_DIR,walker,f"pullx_{len(pullxs)}.npy")
        if os.path.isfile(npy):
            data[walker] = np.load(npy)
        else:
            print(f"Warning {npy} does not seem up-to-date")

    #Plot the walkers distributions
    for walker in data.keys():
        plt.scatter(data[walker][:-1, 0], data[walker][:-1, 1], label=walker, alpha=0.5)
    plt.legend()
    plt.title('Pullx')
    plt.savefig('Pullx.png', dpi=300, facecolor='white')



    #Plot each walker separately
    max_x, min_x = max(data[walker][:-1:100, 0]), min(data[walker][:-1:100, 0])
    max_y, min_y = max(data[walker][:-1:100, 1]), min(data[walker][:-1:100, 1])

    if len(walkers) <= 3:
        fig,ax = plt.subplots(1,len(walkers), figsize=(8*len(walkers), 8), sharey=True)
    else:
        fig,ax = plt.subplots(2,len(walkers), figsize=(8*len(walkers), 8), sharey=True)

    for i, key in enumerate(data.keys()):
        c = next(colors)["color"]
        ax[i].scatter(data[key][::100,0],data[key][::100,1], color=c, label=key)
        ax[i].set_ylim((min_y,max_y))
        ax[i].set_xlim((min_x,max_x))
        ax[i].set_xlabel("CV1", fontsize=22)
        ax[i].legend(fontsize=20)

    if len(walkers) <= 3:
        ax[0].set_ylabel("CV2", fontsize=22)
    else:
        ax[0].set_ylabel("CV2", fontsize=22)
        ax[4].set_ylabel("CV2", fontsize=22)    

    plt.suptitle('Walkers Pullx', fontsize=24)
    fig.tight_layout()
    plt.savefig('Pullx_split.png', dpi=300, facecolor='white')

#Plot AWH edr data
if plot_PMF:
    data = np.loadtxt(os.path.join(WORKING_DIR,awh_xvg), comments=['@','#'])
    fig = plt.figure(clear=True)
    fig,ax = plt.subplots(2,3, figsize=(20,10), sharex=True)

    X = np.asarray(sorted(set(data[:,0])))
    Y = np.asarray(sorted(set(data[:,1])))
    titles = ["PMF", "Coord Bias", "Coord Distribution", " λ distribution",
               "Target Distribution", "Friction"]
    n = 0

    for i in range(2):
        for j in range(3):

            if n == 0:
                Z = copy.deepcopy(data[:,n+2].reshape(X.shape[0],Y.shape[0]))
                Z[np.where(Z > 200)] = np.nan
                Z[np.where(Z == 0)] = np.nan
            else:
                Z = copy.deepcopy(data[:,n+2].reshape(X.shape[0],Y.shape[0]))
                Z[np.where(Z == 0)] = np.nan

            if n <= 1:
                levels_c= 10
                levels_cf = 50
                cmap=cm.jet
            else:
                levels_c = 10
                levels_cf = 10
                cmap=cm.magma_r
            
            ax[i,j].contour(X, Y, Z.T, levels=levels_c, linewidths=0.5, colors='k')
            cntr = ax[i,j].contourf(X,Y, Z.T, levels=levels_cf, cmap=cmap)
            ax[i,j].set_title(titles[n], fontsize=24)

            if j == 0:
                ax[i,j].set_ylabel("CV2", fontsize=20)
            if i == 1:
                ax[i,j].set_xlabel("CV1", fontsize=20)

            if n == 0:
                divider = make_axes_locatable(ax[0,0])
                cax = divider.append_axes('right', size='5%', pad=0.05)
                fig.colorbar(cntr, cax=cax, orientation='vertical')

            n+=1

    fig.tight_layout()
    plt.savefig(os.path.join(WORKING_DIR,"AWH_data_1.png"))

    plt.close(fig)


