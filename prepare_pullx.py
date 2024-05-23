import os
import numpy as np
from multiprocessing import Process

WORKING_DIR = "test/1D"
dim = 1

#Parallelize one walker per each core
def runInParallel(walkers):
  proc = []
  for walker in walkers:
    p = Process(target=concat_xvg, args=(walker,))
    p.start()
    proc.append(p)
  for p in proc:
    p.join()

#Function for xvg reading
def concat_xvg(directory):
    #concatenate all the XVG pullx
    xvglist=[]

    #Ordered list of all pullx files
    lsdir = [ x for x in os.listdir(directory) if x.endswith('pullx.xvg') ]
    sorted_list = sorted(lsdir, key=lambda x: (x.split('.')[1].split('_')[0]))

    #Read xvg and append the array in a list
    if dim == 1 :
    
      for xvg in sorted_list:
          xvglist.append(np.loadtxt(os.path.join(directory, xvg), comments=['#','@'], usecols=[-1]))

      np.save(os.path.join(directory,f"pullx_{len(lsdir)}.npy"), np.concatenate((xvglist), axis=0)) #concatenate all the arrays
    
    elif dim == 2 :
        for xvg in sorted_list:
          xvglist.append(np.loadtxt(os.path.join(directory, xvg), comments=['#','@'], usecols=[-2, -1]))

        np.save(os.path.join(directory,f"pullx_{len(lsdir)}.npy"), np.concatenate((xvglist), axis=0)) #concatenate all the arrays
    
    else:
       print("Postprocessing implemented just for dimensions <= 2")

    return 1

#Main program
if __name__ == '__main__':
  walkers = [ os.path.join(WORKING_DIR, x) for x in os.listdir(WORKING_DIR) if x.startswith('walker') ]
  print(f"Reading and compressing pullx for {walkers}")
  runInParallel(walkers)
