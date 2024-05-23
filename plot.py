import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import ticker, cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import copy
from multiprocessing import Process

matplotlib.use('Agg')

plt.rcParams['xtick.labelsize'] = 20
plt.rcParams['ytick.labelsize'] = 20
plt.rcParams['figure.dpi'] = 300
colors = plt.rcParams["axes.prop_cycle"]()

WORKING_DIR       = "test/2D"
plot_pullx        = False
plot_PMF          = False
SUFFIX            = "walker"
#awh_xvg           = "test/1D/walker_1/awh_data/awh_t183300.xvg"
awh_xvg           = "test/2D/walker_1/awh_data/awh_t183300.xvg"


class Pullx:
    """Pullx postprocessing. It assumes that the walkers folders are called {suffix}.
    Attributes:"""

    def __init__(self, SIM_DIR, suffix):
        
        self.sim_dir = SIM_DIR

        self.walk_dirs = [ x for x in os.listdir(SIM_DIR) if x.startswith(suffix) ]
    
        self.n_walkers = len(self.walk_dirs)

        self.xvg_dict = {}

        self.npy_dict = {}

    def get_xvgdict(self):

        for walker in self.walk_dirs:

            WALKER_DIR = os.path.join( self.sim_dir, walker )

            ls_pullx = [ x for x in os.listdir(WALKER_DIR) if x.endswith('pullx.xvg') ]

            ordered_ls_pullx = sorted( ls_pullx, key=lambda x: (x.split('.')[1].split('_')[0]) )

            self.xvg_dict[walker] = list(ordered_ls_pullx)
    
    def load_npy(self):
            
            for walker in self.walk_dirs:

                WALKER_DIR = os.path.join( self.sim_dir, walker )

                npy = [ x for x in os.listdir(WALKER_DIR) if (x.endswith('npy') and 'pullx' in x) ]

                if len(npy) == 1:
                    self.npy_dict[walker] = np.load(os.path.join(WALKER_DIR, npy[0]))
                else:
                    print("More than one npy found")
                    break

    def concat_xvg(self, walker):
        """Read, concatenate and compress pullx.xvg files"""

        sim_dir = self.sim_dir

        self.get_xvgdict()

        xvg_dict = self.xvg_dict

        xvglist = []

        for xvg in xvg_dict[walker]:

            path_xvg = os.path.join(sim_dir, walker, xvg)

            xvglist.append(np.loadtxt(path_xvg, comments=['#','@'], usecols=[-2, -1]))
        
        np.save(os.path.join(sim_dir, walker,f"pullx_{len(xvglist)}.npy"),
                 np.concatenate((xvglist), axis=0)) #concatenate all in one array and compress

        return 1
    
    def create_npy(self):
        """Function in parallel"""

        proc = []

        for walker in self.walk_dirs:

            p = Process(target=self.concat_xvg, args=(walker,))

            p.start()

            proc.append(p)

        for p in proc:

            p.join()
     

    def one_plot(self):
        #Plot the walkers distributions togheter. Up to two dimensions
        sim_dir = self.sim_dir
        npy_dict = self.npy_dict

        #Check dimensions
        dimensions = npy_dict[list(npy_dict.keys())[0]].shape

        if dimensions[1] == 1:
            for walker in npy_dict.keys():
                
                plt.plot(npy_dict[walker][:-1, 0],
                            label=walker, alpha=0.5)
                
                plt.legend()

                plt.title('Pullx')

                plt.savefig(os.path.join(sim_dir, 'Pullx.png'), dpi=300, facecolor='white')

        elif dimensions[1] == 2:
            for walker in npy_dict.keys():
                
                plt.scatter(npy_dict[walker][:-1, 0], npy_dict[walker][:-1, 1],
                            label=walker, alpha=0.5)
                
                plt.legend()

                plt.title('Pullx')

                plt.savefig(os.path.join(sim_dir, 'Pullx.png'), dpi=300, facecolor='white')

        else:
            print("Plot not possible for dimensions > 2")



    def split_plots(self):
        #Plot each walker separately. 2D case.
        sim_dir = self.sim_dir
        npy_dict = self.npy_dict

        n_walkers = self.n_walkers

        #Retrieve dimensions
        dimensions = npy_dict[list(npy_dict.keys())[0]].shape
 
        #for walker in self.walk_dirs:
            
        #    max_x, min_x = max(npy_dict[walker][:-1:100, 0]), min(npy_dict[walker][:-1:100, 0])
        #    max_y, min_y = max(npy_dict[walker][:-1:100, 1]), min(npy_dict[walker][:-1:100, 1])

        if n_walkers <= 3:
            fig,ax = plt.subplots(1,n_walkers, figsize=(8*n_walkers, 8), sharey=True)

        else:
            fig,ax = plt.subplots(2,n_walkers, figsize=(8*n_walkers, 8), sharey=True)

        #Check dimensions
        if dimensions[1] == 1:

            for i, key in enumerate(npy_dict.keys()):
                c = next(colors)["color"]
                
                ax[i].plot(npy_dict[key][::100,0], color=c, label=key)
                #ax[i].set_ylim((min_y,max_y))
                #ax[i].set_xlim((min_x,max_x))
                ax[i].set_xlabel("CV1", fontsize=22)
                ax[i].legend(fontsize=20)

        elif dimensions[1] == 2:
            
            for i, key in enumerate(npy_dict.keys()):
                c = next(colors)["color"]
                
                ax[i].scatter(npy_dict[key][::100,0],npy_dict[key][::100,1], color=c, label=key)
                #ax[i].set_ylim((min_y,max_y))
                #ax[i].set_xlim((min_x,max_x))
                ax[i].set_xlabel("CV1", fontsize=22)
                ax[i].legend(fontsize=20)

            if n_walkers <= 3:
                ax[0].set_ylabel("CV2", fontsize=22)
            else:
                ax[0].set_ylabel("CV2", fontsize=22)
                ax[4].set_ylabel("CV2", fontsize=22)  

        else:
            print("Plot not possible for dimensions > 2")


        plt.suptitle('Walkers Pullx', fontsize=24)
        fig.tight_layout()
        plt.savefig(os.path.join(sim_dir,'Pullx_split.png'), dpi=300, facecolor='white')

            
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
    plt.savefig(os.path.join(WORKING_DIR,"AWH_data.png"))

    plt.close(fig)

if __name__ == "__main__":

    pullx = Pullx(WORKING_DIR, SUFFIX)

    print(pullx.walk_dirs, pullx.xvg_dict)
    
    try:
        pullx.load_npy()
    except FileNotFoundError:
        pullx.create_npy()
        pullx.load_npy()
    
    pullx.one_plot()
    pullx.split_plots()


