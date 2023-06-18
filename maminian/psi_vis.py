
import numpy as np
import aux_files # has well data.

import sklearny as sy # loads data, creates an "X" and "y"

from matplotlib import patches
from matplotlib.collections import PatchCollection
from matplotlib import pyplot as plt

import copy

domain_patches = [patches.Polygon(v, closed=True) for v in sy.geom.nodes.coords.T[sy.geom.cells.nodes.T, :]]
dp_base = PatchCollection(domain_patches)

#######################################################

def plot_values(myax, values, cmap=plt.cm.viridis, vmin=None, vmax=None, boundarycolor='k'):    
    '''
    Helper function to plot arbitrary scalar fields on the 
    domain using arbitrary color range and colormap.
    
    myax: pyplot axis 
    values: array of numericals, expected shape (1425,)
    cmap: pyplot-compatible colormap instantiator (e.g. plt.cm.viridis)
    vmin, vmax: scalars setting bounds for the colormap.
    boundarycolor: pyplot-compatible color to highlight domain boundary (set to None for no boundary)
    '''
    dp = PatchCollection(domain_patches)
    dp.set_cmap(cmap)
    dp.set_clim(vmin,vmax)
    
    dp.set_array(values)
    myax.add_collection(dp)
    
    if boundarycolor is not None:
        from matplotlib.collections import LineCollection

        kk = sy.geom.faces.nodes[:,range(sy.geom.faces.num_interior, sy.geom.faces.num)]
        
        tensor = np.transpose( sy.geom.nodes.coords[:, kk], [2,1,0] )
        lc = LineCollection(tensor, color=boundarycolor)
        myax.add_collection(lc)

    return dp

######

fig,ax = plt.subplots(2,2, sharex=True, sharey=True, constrained_layout=True)

vmn = sy.ALL_DATA['Psi_y'].min()
vmx = sy.ALL_DATA['Psi_y'].max()
vmg = max(abs(vmn),abs(vmx))
vmn,vmx = -vmg,vmg
#
for i in range(ax.shape[0]):
    for j in range(ax.shape[1]):
        idx = j + i*ax.shape[1]
        que = plot_values( ax[i,j], sy.ALL_DATA['Psi_y'][:,idx], cmap=plt.cm.PRGn, vmin=vmn, vmax=vmx )
        ax[i,j].text(0.05,0.05, r'$\psi_{%i}$'%(idx), fontsize=18,  transform=ax[i,j].transAxes, bbox={'facecolor': 'w', 'edgecolor': 'k'})
        
        # fill background with hatching
        ax[i,j].fill_between([0,1], [1,1], transform=ax[i,j].transAxes, hatch='////', edgecolor='#666', facecolor=[0,0,0,0], zorder=-100)

fig.show()


