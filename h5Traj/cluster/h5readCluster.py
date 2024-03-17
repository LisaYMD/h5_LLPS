import sys
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import h5py
import hdf5plugin
import itertools
from tqdm import tqdm

############### CAUTION:BEFORE USE ###############

# install hdf5plugin, readdy, h5py before use
# activate readdy environment by conda:
#     source activate readdy

##################################################

from .. import h5readTraj

class h5readCluster( h5readTraj ):
    def __init__(self, filename, dim):
        super().__init__(filename, dim)
        # count nonzero value of n_molecule
        self.variables = []
        return None

    ## classify particles into each cluster (can select particle type, more generalized than below)
    def whole_clustering(self, tim):
        mol_total = int(self.limrec[tim,1]-self.limrec[tim,0])
        molecule = np.zeros((mol_total, 6)) ### [global number, xpos, ypos, zpos, class, self.types_num]
        count = 0
        for j in range(self.limrec[tim,0],self.limrec[tim,1]):
            molecule[count,5] = self.rec[j][0] # particle type 
            molecule[count,0] = self.rec[j][1] # global number
            molecule[count,1:4] = self.rec[j][3][:] # positions
            count += 1
        assert count == mol_total
        # read particles
        current = self.limparts[tim,0]
        end = self.limparts[tim,1]
        clust_label = 1
        while current < end:
            progress = int(self.parts[int(current)])
            newlists = self.parts[int(current+1):int(current+progress+1)]
            mol_current = list(set(newlists) & set(list(molecule[:,0])))
            for mol in range(0, len(mol_current)):
                molecule[np.any(np.array([molecule[:,0]])==mol_current[mol], axis=0), 4] = clust_label
            clust_label += 1
            current += int(progress)+1
        return molecule

    ## classify particles into each cluster
    def particle_clustering(self, tim, mol, mol_str, part_in_mol):
        mol_count = -1
        for m in range(0, len(self.mlists)):
            if mol == self.mlists[m]:
                mol_count = self.molcount[m]
        if mol_count == -1 or mol_count == 0:
            print("ERROR: No "+mol+" molecule in this system. Please confirm again.")
            sys.exit()
        #particle_label = self.types_num[self.types_str.index(mol_str)]
        particle_names = [s for s in self.types_str if mol_str in s]
        particle_label = []
        for p in range(0, len(particle_names)):
            particle_label.append(self.types_num[self.types_str.index(particle_names[p])])
        particle = np.zeros((int(mol_count*part_in_mol), 5)) # [global number(needs imput), x position, y position, z position, class] 
        count = 0
        for j in range(self.limrec[tim,0], self.limrec[tim,1]):
            if self.rec[j][0] in particle_label:
                particle[count,0] = self.rec[j][1]
                particle[count,1] = self.rec[j][3][0]
                particle[count,2] = self.rec[j][3][1]
                particle[count,3] = self.rec[j][3][2]
                count += 1
        assert count == int(mol_count*part_in_mol)
        # read particles
        current = self.limparts[tim, 0]
        end = self.limparts[tim, 1]
        clust_label = 1
        while current < end:
            progress = int(self.parts[int(current)])
            newlists = self.parts[int(current+1):int(current+progress+1)]
            particle_current = list(set(newlists) & set(list(particle[:,0])))
            for amp in range(0, len(particle_current)):
                particle[np.any(np.array([particle[:,0]])==particle_current[amp], axis=0), 4] = clust_label
            clust_label += 1
            current += int(progress)+1
        return particle    

