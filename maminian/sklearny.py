
from sklearn import linear_model
import numpy as np
from matplotlib import pyplot as plt
import h5py

import aux_files

import sys
import os
sys.path.append(os.path.abspath('../'))


from sdfs.geom_mrst import geom_plot, GeomMRST

'''
Manuch --

I want to do an extremely naive linear regression 
of observed inputs/outputs with randomized 
train/test sets via simple random selection with 
replacement.
'''

geom_filename = '../data/geom_1x.mat'
geom = GeomMRST(geom_filename)

def load():
    with h5py.File('../data/mpi23_ens_data.h5','r') as f:
        output = {
            "xi_ens"   : f['xi_ens'][:],
            "ytms_ens" : f['ytms_ens'][:],
            "u_ens"    : f['u_ens'][:],
            "yref"     : f['yref'][:],
            "ypred"    : f['ypred'][:],
            "Psi_y"    : f['Psi_y'][:],
            "ytm"      : f['ytm'][()],
            "Nens"     : f['Nens'][()],
            "Nxi"      : f['Nxi'][()]
            }
    return output

def vis(quantity, figax=None):
    
    # via data_read_example
    if figax is None:
        fig, ax = plt.subplots()
    else:
        fig, ax = figax
    plot = geom_plot(geom, quantity, ax)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.autoscale(tight=True)
    fig.colorbar(plot, ax=ax)
    #fig.tight_layout()
    if figax is None:
        return fig,ax
    else:
        return None
    


# ridge regression; specific type of regularized linear regression
# swap for anything.
#
# For now, investigate xi predicting u

ALL_DATA = load()

#
nreps = 1
#fig,ax = plt.subplots(3,4, sharex=True, sharey=True)



#test_fraction = 0.1 # A fraction; gets rounded to integer later.

# for now - xi predicts u
# INPUT VAR
in_var = "xi_ens"
# OUTPUT
out_var = "u_ens"

# figures these out when data is loaded.
dim_in = ALL_DATA[in_var].shape[1]  # dimensionality of input data
dim_out = ALL_DATA[out_var].shape[1] # dimensionality of output data.
nobs = ALL_DATA[in_var].shape[0]   # total number of input/output pairs.

#


####

X = ALL_DATA[in_var]
y = ALL_DATA[out_var]

####
# Everything above this line is what I want as a module-like thing.
####
if __name__=="__main__":
    
    subs = 1000
    X_sub = X[:subs,:]
    y_sub = y[:subs,:]
    
    #trainsize = int( (1-test_fraction)*nobs )
    trainsize = X_sub.shape[0] - 1
    testsize = X_sub.shape[0] - trainsize
    
    orders = [np.random.permutation(X_sub.shape[1])]
    models = [ linear_model.Ridge() for _ in range(nreps) ]
    test_train_sets = [[oi[:trainsize], oi[trainsize:]] for oi in orders]
    
    
    
    preds = []
    
    for i,(train_idx,test_idx) in enumerate(test_train_sets):
        model = models[i]
        
        X_train = X_sub[train_idx]
        y_train = y_sub[train_idx]
        
        X_test = X_sub[test_idx]
        y_test = y_sub[test_idx]
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        #err = y_pred - y_test
        preds.append(y_pred)
        
        models[i] = model # just in case...
    #
    
    # visualize example output
    fig1,ax1 = vis(y_pred[0])
    # visualize example error
    fig2,ax2 = vis(y_pred[0] - y_test[0])
    
    aux_files.overlay_wells(geom, ax=ax1, c='r')
    aux_files.overlay_wells(geom, ax=ax2, c='r')
    
