import numpy as np 
import polychrom.polymerutils
import polychrom.polymer_analyses
from polychrom.polymer_analyses import contact_scaling as cs
from multiprocessing import Pool 
from scipy.ndimage.filters import gaussian_filter1d


def get_cp(filename):
        
    N=22545
    
    data = polychrom.polymerutils.load(filename)
    bins = np.array(polychrom.polymer_analyses.generate_bins(N, start=6, bins_per_order_magn=10))
    
    mids, cp = cs(data, bins, cutoff=CUTOFF)
    
    return mids, cp


folder = "simulations/"
filenames = list_URIs(folder)
with Pool(30) as mypool:
    cp_array = mypool.map(get_cp, filenames)
bins, cp = np.mean(cp_array, axis=0)


# log-derivative
diff = gaussian_filter1d((np.log10(cp[1:])-np.log10(cp[:-1]))/(np.log10(bins[1:])-np.log10(bins[:-1])), sg=1.5, mode='nearest')
