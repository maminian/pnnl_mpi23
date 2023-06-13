import sys
#import path
import h5py
import numpy as np


data_path = '../data/'
geom_filename = data_path + f'geom_1x.mat'
bc_filename = data_path + f'bc_1x.mat'
conduct_filename = data_path + f'conduct_log_1x.mat'
well_cells_filename = data_path + f'well_cells_1x.mat'
yobs_filename = data_path + f'yobs_200_1x.npy'

with h5py.File(conduct_filename, 'r') as f:
    ytrue = f.get('conduct_log')[:].ravel()

with h5py.File(well_cells_filename, 'r') as f:
    iuobs = f.get('well_cells')[:].ravel() - 1

iyobs = np.load(yobs_filename)

with h5py.File(well_cells_filename, 'r') as f:
    iuobs = f.get('well_cells')[:].ravel() - 1
    
def overlay_wells(geom, ax, c='r'):
    x,y = geom.cells.centroids[:,iuobs]
    ax.scatter(x,y, c=c, marker='x')
    return
    