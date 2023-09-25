import sys
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import h5py
import hdf5plugin
import itertools
from tqdm import tqdm
import time

############### CAUTION:BEFORE USE ###############

# install hdf5plugin, readdy, h5py before use
# activate readdy environment by conda:
#     source activate readdy

##################################################

def size(aa):
    radius = 0.1*2.24*(aa **0.392)
    return radius 

class h5readCluster: 
    def __init__(self, filename):
        f = h5py.File(filename, 'r')
        self.filename = filename
        self.traj = f['readdy']['trajectory']
        self.limrec = self.traj['limits']
        self.rec = self.traj['records']
        self.topo = f['readdy']['observables']['topologies']
        self.limparts = self.topo['limitsParticles']
        self.parts = self.topo['particles']
        types = f['readdy']['config']['particle_types']
        self.types_str = [types[k][0].decode() for k in range(0, len(types))] 
        self.types_num = [types[k][1] for k in range(0, len(types))]
        info = np.atleast_1d(f['readdy']['config']['general'][...])[0].decode()
        boxinfo = info[(info.find('"box_size":[')+len(str('"box_size":['))):(info.find("],"))].split(",")
        self.xbox = np.array(boxinfo).astype("float")[0]
        self.zbox = np.array(boxinfo).astype("float")[2]
        self.size_list = []
        # constant values 
        aa_ampar = 936 # DsRed # of residues (pseudo-ampar complex)
        aa_tarp = 211
        aa_ntd = 5
        aa_pdz = 86
        aa_sh3 = 70
        aa_gk = 165
        aa_glun2bc = 256 # residues 1226–1482, GluN2Bc
        #aa_nmdar = 241 # monomer of eqFP670 (dimeric FP: pseudo-nmdar complex)
        aa_nmdar = 482
        camk2_hub = 11/2 # nanometer radius of camk2 hub complex
        camk2_kinase = 4.5/2 # nanometer radius of camk2 kinase domain
        camk2_linker = 3. # nanometer length of camk2 linker between hub and kinase
        for l in self.types_str:
            if l == "A":
                self.size_list.append(size(aa_ampar))
            elif l == "B0" or l == "B1":
                self.size_list.append(size(aa_tarp))
            elif l == "C":
                self.size_list.append(size(aa_ntd))
            elif l == "D01" or l == "D02" or l == "D03" or l == "D11" or l == "D12" or l == "D13":
                self.size_list.append(size(aa_pdz))
            elif l == "E":
                self.size_list.append(size(aa_sh3))
            elif l == "F":
                self.size_list.append(size(aa_gk))
            elif l == "G":
                self.size_list.append(size(aa_nmdar))
            elif l in ["N00", "N01", "N10", "N11"]:
                self.size_list.append(size(aa_glun2bc))
            elif l == "M":
                self.size_list.append(camk2_hub)
            elif l == "K0" or l == "K1":
                self.size_list.append(camk2_kinase)
            else:
                print("ERROR: Unclassified particle type")
        # count the number of molecules at initial state
        ini = self.rec[self.limrec[0, 0]:self.limrec[0, 1]]
        self.n_ampar, self.n_psd95, self.n_nmdar, self.n_camk2 = 0, 0, 0, 0
        self.mlists = []
        if 'A' in self.types_str:
            self.n_ampar=len([ini[r] for r in range(0, len(ini)) if (ini[r][0]==self.types_num[self.types_str.index('A')])]) 
            self.mlists.append("AMPAR")
        if 'F' in self.types_str:
            self.n_psd95=len([ini[r] for r in range(0, len(ini)) if (ini[r][0]==self.types_num[self.types_str.index('F')])])
            self.mlists.append("PSD-95")
        if 'G' in self.types_str:
            self.n_nmdar=len([ini[r] for r in range(0, len(ini)) if (ini[r][0]==self.types_num[self.types_str.index('G')])]) 
            self.mlists.append("NMDAR")
        if 'M' in self.types_str:
            self.n_camk2=len([ini[r] for r in range(0, len(ini)) if (ini[r][0]==self.types_num[self.types_str.index('M')])]) 
            self.mlists.append("CaMKII")
        self.molcount = [self.n_ampar, self.n_psd95, self.n_nmdar, self.n_camk2]
        self.moltype = [["A", ["B0", "B1"]], ["C", ["D01","D11"], ["D02", "D12"], ["D03", "D13"], "E", "F"], ["G", ["N00", "N01", "N10", "N11"]], ["M", ["K0", "K1"]]]
        # count nonzero value of n_molecule
        self.variables = []

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
        if mol_count == -1:
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

#result = h5readCluster("PSD4_slab_activated.h5")
#print(np.unique(result.whole_clustering(5000)[:,4]))
#print(result.particle_clustering(5000, "PSD-95", "D", 3)[:,4])
