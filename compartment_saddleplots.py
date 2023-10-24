from multiprocessing import Pool 
import numpy as np 
import polychrom.polymerutils
import polychrom.polymer_analyses
import matplotlib.pyplot as plt
from scipy.sparse.linalg import eigs
from mirnylib.numutils import observedOverExpected

def get_saddle(m, qrange=(0.02,0.98), n_bins=50, min_diag=200, max_diag=5000):
    
    vals, vecs = eigs(m, k=2, which='LR')

    track_values = np.real(vecs[:, 1])
     
    qlo, qhi = qrange
    q_edges = np.linspace(qlo, qhi, n_bins + 1)*100
    binedges = np.nanpercentile(track_values, q_edges)

    digitized = np.digitize(track_values, binedges, right=False)

    matrix=m.copy()
    for s in range(max_diag,len(matrix)):
        for i in range(len(matrix)-s):
            matrix[i,i+s]=np.nan
            matrix[i+s,i]=np.nan

    for s in range(min_diag):
        for i in range(len(matrix)-s):
            matrix[i,i+s]=np.nan
            matrix[i+s,i]=np.nan

    S = np.zeros((n_bins, n_bins))
    C = np.zeros((n_bins, n_bins))
    
    for i in range(1,n_bins+1):
        row_mask = digitized == i
        for j in range(1,n_bins+1):
            col_mask = digitized == j
            data = matrix[row_mask, :][:, col_mask]
            data = data[np.isfinite(data)]
            S[i-1, j-1] = np.sum(data)
            C[i-1, j-1] = float(len(data))

    intra=(np.sum(S[0:10,0:10])+np.sum(S[-10:-1,-10:-1]))/(np.sum(C[0:10,0:10])+np.sum(C[-10:-1,-10:-1]))
    inter=(np.sum(S[0:10,-10:-1])+np.sum(S[-10:-1,0:10]))/(np.sum(C[0:10,-10:-1])+np.sum(C[-10:-1,0:10]))

            
    return S, C, intra/inter


def binned_hmap(binsize, hmap):
    
    n = len(hmap)
    n1 = int(n/binsize)
    hmap1 = np.zeros((n1, n1))
    
    for i in range(n1):
        for j in range(i, n1):
            hmap1[i,j] = np.mean(hmap[i*binsize:(i+1)*binsize, j*binsize:(j+1)*binsize])
            hmap1[j,i] = hmap1[i,j]
            
    return hmap1




# Binning into 100 beads = 400kb
hmap = binned_hmap(100, hmap_raw)

hmap[hmap==0]=1

m=observedOverExpected(hmap)

S1_0 = np.zeros((50,50))
C1_0 = np.zeros((50,50))
strens = []
for i in range(10):
    
    # scale range from 800kb to 20Mb
    S1, C1, stren = get_saddle(m, min_diag=2,max_diag=50)
    S1_0 += S1/10
    C1_0 += C1/10
    strens.append(stren)
    
im=plt.imshow(np.log10(S1_0/C1_0), cmap='coolwarm', vmin=-0.5, vmax=0.5)
plt.colorbar(im,fraction=0.046, pad=0.04)
plt.xticks([])
plt.yticks([])
plt.show()

# The mean compartmental score
print(strens)




S1_0 = np.zeros((50,50))
C1_0 = np.zeros((50,50))
strens = []
for i in range(10):
    
    # scale range from 20Mb to 80Mb
    S1, C1, stren = get_saddle(m, min_diag=50,max_diag=200)
    S1_0 += S1/10
    C1_0 += C1/10
    strens.append(stren)
    
im=plt.imshow(np.log10(S1_0/C1_0), cmap='coolwarm', vmin=-0.5, vmax=0.5)
plt.colorbar(im,fraction=0.046, pad=0.04)
plt.xticks([])
plt.yticks([])
plt.show()

# The mean compartmental score
print(strens)
