
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
    dp = PatchCollection(domain_patches)
    dp.set_cmap(cmap)
    dp.set_clim(vmin,vmax)
    
    dp.set_array(values)
    myax.add_collection(dp)
    
    #boundarycolor = 'k' if boundarycolor is None else boundarycolor
    
    # todo: linecollection
    # TODO: ... pretty sure the sequence of data structure 
    # ops are all ok ... ???
    # Are we sure the algo identifying exterior faces is what it should be?

    if boundarycolor is not None:
        for i in range(sy.geom.faces.num_interior, sy.geom.faces.num):
            
            #node_idxs = sy.geom.faces.nodes[:,i]
            kk = sy.geom.faces.nodes[:,i]
            #bdry = any( sy.geom.faces.neighbors[:,i] < 0 )
            
            xx,yy = sy.geom.nodes.coords[:, kk]
            myax.plot( xx, yy, c=boundarycolor )
    
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
        ax[i,j].text(0,0, r'$\psi_{%i}$'%(idx), transform=ax[i,j].transAxes)


fig.show()


