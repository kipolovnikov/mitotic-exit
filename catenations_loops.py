import numpy as np 
import polychrom.polymerutils
import matplotlib.pyplot as plt
from polychrom.polymer_analyses import getLinkingNumber
from polychrom.hdf5_format import HDF5Reporter, list_URIs, load_URI, load_hdf5_file, save_hdf5_file


def get_linking_number(file):
    
    data = polychrom.polymerutils.load(file)
    
    loops=load_URI(file)['loops']

    ln = np.zeros((len(loops), len(loops)))
    for i in range(len(loops)):
        for j in range(i+1,len(loops)):
            ln[i,j]=getLinkingNumber(data[loops[i,0]:loops[i,1]], data[loops[j,0]:loops[j,1]], randomOffset=False)
            ln[j,i]=ln[i,j]

    z=np.mean(np.array([np.count_nonzero(ln[:, i]) for i in range(len(ln))]))
    
    return ln, z


folder = "simulations/"        
filenames = list_URIs(folder)

ln, z = get_linking_number(filenames[-1])

# mean number of catenations per loop
print(z)

# a matrix of the pairwise linking numbers
plt.spy(ln)
plt.show()
