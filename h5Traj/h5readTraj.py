import os
import sys
import numpy as np
import itertools
import readdy
import h5py
import matplotlib.pyplot as plt
from tqdm import tqdm

############### CAUTION:BEFORE USE ###############

# install hdf5plugin, readdy, h5py before use
# activate readdy environment by conda:
#     source activate readdy

##################################################

def size(aa):
    radius = 0.1*2.24*(aa **0.392)
    return radius

def diff_const(aa, string):
    rh = 1.45* size(aa) # radius of hydration (nanometer)
    # stokes-einstein expression
    RT = system.kbt.magnitude # kilojoule/mole = 10^(3)*Neuton*Meter/mole = 10^(3+9)*Neuton*nanometer/mole
    # Pascal = Neuton * (Meter)**(-2) = Neuton * nanometer^-2 * 10^(18)
    if string == "cell":
        eta = 0.85137 # milliPascal*second = 10^-3 * Neuton*nanometer^-2*10^(18)*nanosecond*10^9
    elif string == "lipid":
        eta = 0.85137*0.1 # milliPascal*second = 10^-3 * Neuton*nanometer^-2*10^(18)*nanosecond*10^9
    else:
        None
    diff = RT*10/(6.02214076*6*np.pi*eta*rh)
    return diff

class h5readTraj:
    def __init__(self, filename):
        f = h5py.File(filename, 'r')
        self.filename = filename
        traj = f['readdy']['trajectory']
        types = f['readdy']['config']['particle_types']
        self.types_str = [types[k][0].decode() for k in range(0, len(types))] 
        self.types_num = [types[k][1] for k in range(0, len(types))]
        info = np.atleast_1d(f['readdy']['config']['general'][...])[0].decode()
        self.limrec = traj['limits']
        self.rec = traj['records']
        ini = self.rec[self.limrec[0, 0]:self.limrec[0, 1]] 
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
        self.mol_detail = [["A", "B0", "B1"], ["C", "D01", "D02", "D03", "D11", "D12", "D13", "E", "F"], ["G", "N00", "N01", "N10", "N11"], ["M", "K0", "K1"]]
        self.mcount = [self.n_ampar, self.n_psd95, self.n_nmdar, self.n_camk2]

    def viewTraj(self):
        # size of each domain is determined: 
        trajectory = readdy.Trajectory(self.filename)
        radii_dict = {}
        color_dict = {}
        for r in range(0, len(self.types_str)):
            radii_dict[self.types_str[r]] = self.size_list[r]
            if self.types_str[r] == "A":
                color_dict[self.types_str[r]] = 1
            elif self.types_str[r][0] == "B":  ## several
                color_dict[self.types_str[r]] = 8
            elif self.types_str[r] == "C":
                color_dict[self.types_str[r]] = 10
            elif self.types_str[r][0] == "D":  ## several
                color_dict[self.types_str[r]] = 3
            elif self.types_str[r] == "E":
                color_dict[self.types_str[r]] = 7
            elif self.types_str[r] == "F":
                color_dict[self.types_str[r]] = 0
            elif self.types_str[r] == "G":
                color_dict[self.types_str[r]] = 4
            elif self.types_str[r][0] == "N":  ## several
                color_dict[self.types_str[r]] = 9
            elif self.types_str[r] == "M":
                color_dict[self.types_str[r]] = 2
            elif self.types_str[r][0] == "K":  ## several
                color_dict[self.types_str[r]] = 5
            else:
                print("ERROR: Unclassified particle type")
        trajectory.convert_to_xyz(particle_radii=radii_dict, color_ids=color_dict, draw_box=True)
        return None

    def viewCheckpoint(self):
        # read checkpoint
        custom_unit = {"length_unit":"nanometer", "time_unit":"nanosecond", "energy_unit":"kilojoule/mol"}
        system = readdy.ReactionDiffusionSystem([self.xbox, self.xbox, self.zbox], temperature=300.*readdy.units.kelvin, periodic_boundary_conditions=[True, True, True], unit_system=custom_unit)
        # write settings here
        ckpt = readdy.Trajectory(self.filename)
        radii_dict = {}
        color_dict = {}
        for r in range(0, len(self.types_str)):
            radii_dict[self.types_str[r]] = self.size_list[r]
            if self.types_str[r] == "A":
                color_dict[self.types_str[r]] = 1
            elif self.types_str[r][0] == "B":  ## several
                color_dict[self.types_str[r]] = 8
            elif self.types_str[r] == "C":
                color_dict[self.types_str[r]] = 10
            elif self.types_str[r][0] == "D":  ## several
                color_dict[self.types_str[r]] = 3
            elif self.types_str[r] == "E":
                color_dict[self.types_str[r]] = 7
            elif self.types_str[r] == "F":
                color_dict[self.types_str[r]] = 0
            elif self.types_str[r] == "G":
                color_dict[self.types_str[r]] = 4
            elif self.types_str[r][0] == "N":  ## several
                color_dict[self.types_str[r]] = 9
            elif self.types_str[r] == "M":
                color_dict[self.types_str[r]] = 2
            elif self.types_str[r][0] == "K":  ## several
                color_dict[self.types_str[r]] = 5
            else:
                print("ERROR: Unclassified particle type")
        ckpt.convert_to_xyz(particle_radii=radii_dict, color_ids=color_dict, draw_box=True)
        return None

    def radial_distribution_function(self, part_str, part_count, inital_time, duration):
        max_radius = 50
        radius_slice = 0.1
        radius_lists = np.arange(0, max_radius, radius_slice)
        coor_list = [-1, 0, 1]
        mol_size = self.size_list[self.types_str.index(part_str)]
        #print(mol_size)
        hist_accuml = np.zeros(radius_lists.shape[0]-1)
        hist_accuml_partave = np.zeros(radius_lists.shape[0]-1)
        hist_accuml_timeave = np.zeros(radius_lists.shape[0]-1)
        for tim in tqdm(range(initial_time,int(initial_time+duration))):
            # make a group with the same particles
            part_num = self.types_num[self.types_str.index(part_str)]
            for p in range(0, 4):
                if part_str in self.mol_detail[p]:
                    mol_count = self.mcount[p]
            new_part = np.zeros([int(mol_count*part_count), 4])
            count = 0
            for l in range(self.limrec[tim,0], self.limrec[tim,1]):
                if self.rec[l][0] == part_num:
                    new_part[count,0] = self.rec[l][1]
                    new_part[count,1:4] = self.rec[l][3]
                    count += 1
            assert count == mol_count*part_count
            # measure the distance between them from one representative particle
            for p in range(0, len(new_part)):
                another_part = np.zeros([new_part.shape[0]*27, new_part.shape[1]])
                for x_ext in range(0, len(coor_list)):
                    for y_ext in range(0, len(coor_list)):
                        for z_ext in range(0, len(coor_list)):
                            c = int(x_ext + y_ext*3 + z_ext*9)
                            start = int(c*len(new_part))
                            end = int((c+1)*len(new_part))
                            another_part[start:end,:] = new_part.copy()
                            another_part[start:end,1] += coor_list[x_ext]*self.xbox
                            another_part[start:end,2] += coor_list[y_ext]*self.xbox
                            another_part[start:end,3] += coor_list[z_ext]*self.zbox
                copied_part = new_part.copy()
                copied_part[:,0] = 0
                dist = another_part[:,:]-copied_part[p,:]
                dist_wotgt = dist[np.any(np.array([dist[:,0]])!=new_part[p,0], axis=0)]
                distance = np.zeros([dist_wotgt.shape[0], 2])
                distance[:,0] = dist_wotgt[:,0]
                distance[:,1] = np.sqrt(dist_wotgt[:,1]**2+dist_wotgt[:,2]**2+dist_wotgt[:,3]**2)
                dist_final = np.zeros(int(distance.shape[0]/27))
                for d in range(0, len(dist_final)):
                    target = distance[np.any(np.array([distance[:,0]])==distance[d,0], axis=0)]
                    dist_final[d] = np.amin(target[:,1])
                for r in range(0, len(radius_lists)-1):
                    minlim = radius_lists[r] - mol_size
                    maxlim = radius_lists[r+1] + mol_size
                    in_range = dist_final[(dist_final>radius_lists[r]-mol_size)&(dist_final<radius_lists[r+1]+mol_size)]
                    hist_accuml[r] = len(in_range)
                hist_accuml_partave += hist_accuml/len(new_part)
            hist_accuml_timeave += hist_accuml_partave/duration
        gr = hist_accuml_timeave/(4*radius_slice*np.pi*(radius_lists[1:])**2)
        return radius_lists[1:], gr

    def plot_RDF(self, part_str, part_count, start, duration):
        rad, gr = self.radial_distribution_function(self, part_str, part_count, start, duration)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(rad, gr)
        plt.savefig("radial_distribution_"+part_str+".png")
        plt.show()
        return None

    def relative_distance_from_COM(self, part_str, part_count, tim, com, max_radius=100, radius_slice=0.1):
        radius_lists = np.arange(0, max_radius, radius_slice)
        coor_list = [-1, 0, 1]
        mol_size = self.size_list[self.types_str.index(part_str)]
        hist_accuml = np.zeros(radius_lists.shape[0]-1)
        hist_accuml_partave = np.zeros(radius_lists.shape[0]-1)
        # make a group with the same particles
        part_num = self.types_num[self.types_str.index(part_str)]
        for p in range(0, 4):
            if part_str in self.mol_detail[p]:
                mol_count = self.mcount[p]
        new_part = np.zeros([int(mol_count*part_count), 4])
        count = 0
        for l in range(self.limrec[tim,0], self.limrec[tim,1]):
            if self.rec[l][0] == part_num:
                new_part[count,0] = self.rec[l][1]
                new_part[count,1:4] = self.rec[l][3]
                count += 1
        assert count == mol_count*part_count
        # measure the distance between them from one representative particle
        for p in range(0, len(new_part)):
            another_part = np.zeros([new_part.shape[0]*27, new_part.shape[1]])
            for x_ext in range(0, len(coor_list)):
                for y_ext in range(0, len(coor_list)):
                    for z_ext in range(0, len(coor_list)):
                        c = int(x_ext + y_ext*3 + z_ext*9)
                        start = int(c*len(new_part))
                        end = int((c+1)*len(new_part))
                        another_part[start:end,:] = new_part.copy()
                        another_part[start:end,1] += coor_list[x_ext]*self.xbox
                        another_part[start:end,2] += coor_list[y_ext]*self.xbox
                        another_part[start:end,3] += coor_list[z_ext]*self.zbox
            center_of_mass = np.zeros([1,4])
            center_of_mass[0,1:4] = com
            dist = another_part[:,:]-center_of_mass
            distance = np.zeros([dist.shape[0], 2])
            distance[:,0] = dist[:,0]
            distance[:,1] = np.sqrt(dist[:,1]**2+dist[:,2]**2+dist[:,3]**2)
            dist_final = np.zeros(int(distance.shape[0]/27))
            for d in range(0, len(dist_final)):
                target = distance[np.any(np.array([distance[:,0]])==distance[d,0], axis=0)]
                dist_final[d] = np.amin(target[:,1])
            for r in range(0, len(radius_lists)-1):
                minlim = radius_lists[r] - mol_size
                maxlim = radius_lists[r+1] + mol_size
                in_range = dist_final[(dist_final>radius_lists[r]-mol_size)&(dist_final<radius_lists[r+1]+mol_size)]
                hist_accuml[r] = len(in_range)
            hist_accuml_partave += hist_accuml/len(new_part)
        relative_distance = hist_accuml_partave
        return radius_lists[1:], relative_distance

