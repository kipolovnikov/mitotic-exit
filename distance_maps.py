from multiprocessing import Pool 
import numpy as np 
import polychrom.polymerutils
import polychrom.polymer_analyses
import matplotlib.pyplot as plt
from polychrom.hdf5_format import HDF5Reporter, list_URIs, load_URI, load_hdf5_file, save_hdf5_file

def get_distance_map(filename):
    
    data = polychrom.polymerutils.load(filename)
    data1=binned_data(data, 10)
    covmat=np.matmul(data1, data1.T)

    cov1=covmat.copy()
    for i in range(len(cov1)):
        cov1[i, :] = covmat[i,i]*np.ones(len(cov1))

    cov2=covmat.copy()
    for i in range(len(cov2)):
        cov2[:, i] = covmat[i,i]*np.ones(len(cov2))

    distmap=-2*covmat+cov1+cov2

    return distmap

def binned_data(data, binsize):
    
    n = len(data)
    n1 = int(n/binsize)
    data1 = np.zeros((n1, 3))
    
    for i in range(n1):
        data1[i,:] = np.mean(data[i*binsize:(i+1)*binsize, :], axis=0)
            
    return data1



folder = "simulations/"        
filenames = list_URIs(folder)

with Pool(30) as mypool:
    mat_arr = mypool.map(get_distance_map, filenames)
distmap = np.mean(mat_arr, axis=0)
