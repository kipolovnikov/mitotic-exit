import numpy as np 
from scipy.spatial import ckdtree

from polychrom.contactmaps import binnedContactMap
import polychrom.polymerutils
import polychrom.polymer_analyses


def customLoadFunction(filename):

    data = polychrom.polymerutils.load(filename)

    return data

def get_cont(a1, a2):
    
    cont = np.zeros((2, 2))
    cont[0, 0] = np.sum(a2[0])
    cont[0, 1] = np.sum(a1[0]-a2[0])
    cont[1, 0] = cont[0, 1]
    cont[1, 1] = cont[0, 0]
    return cont/cont[0, 0]

def customLoadFunctionScale(filename):

    data = polychrom.polymerutils.load(filename)

    global BOX_SIZE
    
    a=BOX_SIZE
    data1=np.mod(data+np.array([a/2, a/2, a/2]), a)
    
    return data1

def customContactFinder(data, cutoff=5):
    
    if data.shape[1] != 3:
        raise ValueError("Incorrect polymer data shape. Must be Nx3.")

    if np.isnan(data).any():
        raise RuntimeError("Data contains NANs")
    
    global CUTOFF
    
    cutoff=CUTOFF

    global BOX_SIZE

    tree = ckdtree.cKDTree(data, boxsize=BOX_SIZE)
    pairs = tree.query_pairs(cutoff, output_type="ndarray")
    
    #print(pairs)
    
    return pairs
    
    
def get_cont_matrix(filenames, ctf=5, binSize=100, n=40):
        
        
    global BOX_SIZE

    BOX_SIZE=40
    
    global CUTOFF
    
    CUTOFF=ctf
    
    a1 = binnedContactMap(filenames, binSize=binSize,  n=n, cutoff=CUTOFF,
                         loadFunction=customLoadFunctionScale, contactFinder=customContactFinder)
    a2 = binnedContactMap(filenames, binSize=binSize,  n=n, cutoff=CUTOFF,
                         loadFunction=customLoadFunction, contactFinder=polychrom.polymer_analyses.calculate_contacts)
    cont = get_cont(a1, a2)

    return cont


folder = "simulations/"
filenames = list_URIs(folder)

#cutoff=3
a=np.array(get_cont_matrix(filenames, ctf=3, n=20))
print(1/a[0,1])

#cutoff=5
a=np.array(get_cont_matrix(filenames, ctf=5, n=20))
print(1/a[0,1])
