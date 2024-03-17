import os
import sys
import numpy as np
import itertools
import readdy
import h5py
import matplotlib.pyplot as plt
from tqdm import tqdm
import json

############### CAUTION:BEFORE USE ###############

# install hdf5plugin, readdy, h5py before use
# activate readdy environment by conda:
#     source activate readdy

##################################################

class h5readTraj:
    def __init__(self, filename, dim=3):
        f = h5py.File(filename, 'r')
        self.filename = filename
        traj = f['readdy']['trajectory']
        types = f['readdy']['config']['particle_types']
        self.types_str = [types[k][0].decode() for k in range(0, len(types))] 
        self.types_num = [types[k][1] for k in range(0, len(types))]
        self.diff_const = [types[k][2] for k in range(0, len(types))]
        info = np.atleast_1d(f['readdy']['config']['general'][...])[0].decode()
        self.limrec = traj['limits']
        self.rec = traj['records']
        self.topo = f['readdy']['observables']['topologies']
        self.limparts = self.topo['limitsParticles']
        self.parts = self.topo['particles'] 
        ini = self.rec[self.limrec[0, 0]:self.limrec[0, 1]] 
        boxinfo = info[(info.find('"box_size":[')+len(str('"box_size":['))):(info.find("],"))].split(",")
        self.xbox = np.array(boxinfo).astype("float")[0]
        self.zbox = np.array(boxinfo).astype("float")[2]
        self.size_list = [] # read from setting file
        self.color_list = []
        self.dimension = dim
        # load setting file here
        dirname = os.path.dirname(__file__)
        if self.dimension == 3:
            filename = os.path.join(dirname, "../mol_settings/3d_settings.json")
            with open(filename) as f:
                settings = json.load(f)
        elif self.dimension == 2:
            filename = os.path.join(dirname, "../mol_settings/2d_settings.json")
            with open(filename) as f:
                settings = json.load(f)
        else:
            print("Unexpected dimension type")
        self.mlists =  settings["molecule"]
        self.molcompose = []
        for l in self.types_str:
            for m in range(0, len(self.mlists)):
                cand = settings["particle_type"][self.mlists[m]]
                if l in cand:
                    self.size_list.append(settings["size"][self.mlists[m]][cand.index(l)])
                    self.color_list.append(settings["color"][self.mlists[m]][cand.index(l)])
                    break
        assert len(self.types_str) == len(self.size_list)
        self.molcount = np.zeros(len(self.mlists))
        self.molchar = []
        for mol in range(0, len(self.mlists)):
            self.molcompose.append(settings["particle_type"][self.mlists[mol]])
            char = settings["particle_type"][self.mlists[mol]][0]
            if char in self.types_str:
                self.molcount[mol]=len([ini[r] for r in range(0, len(ini)) if (ini[r][0]==self.types_num[self.types_str.index(char)])])
                self.molchar.append(char)
        return None
 
    def viewTraj(self):
        # size of each domain is determined: 
        trajectory = readdy.Trajectory(self.filename)
        radii_dict = {}
        color_dict = {}
        for r in range(0, len(self.types_str)):
            radii_dict[self.types_str[r]] = self.size_list[r]
            color_dict[self.types_str[r]] = self.color_list[r] 
        trajectory.convert_to_xyz(particle_radii=radii_dict, color_ids=color_dict, draw_box=True)
        return None

    def radial_distribution_function(self, part_str, part_count, initial_time, duration):
        max_radius = 50
        radius_slice = 0.1
        radius_lists = np.arange(0, max_radius, radius_slice)
        coor_list = [-1, 0, 1]
        mol_size = self.size_list[self.types_str.index(part_str)]
        hist_accuml = np.zeros(radius_lists.shape[0]-1)
        hist_accuml_partave = np.zeros(radius_lists.shape[0]-1)
        hist_accuml_timeave = np.zeros(radius_lists.shape[0]-1)
        for tim in tqdm(range(initial_time,int(initial_time+duration))):
            # make a group with the same particles
            part_num = self.types_num[self.types_str.index(part_str)]
            for p in range(0, len(self.mlists)):
                if part_str in self.molcompose[p]:
                    mol_count = self.molcount[p]
                    break
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

    def radial_distribution_function_diff(self, part_str1, part_count1, part_str2, part_count2, initial_time, duration):
        max_radius = 50
        radius_slice = 0.1
        radius_lists = np.arange(0, max_radius, radius_slice) # define plot range
        coor_list = [-1, 0, 1]
        hist_accuml_timeave = np.zeros(radius_lists.shape[0]-1)
        part_num1 = self.types_num[self.types_str.index(part_str1)]
        part_num2 = self.types_num[self.types_str.index(part_str2)] 
        # check the distribution and choose an axis of deletion
        if self.dimension == 2:
            samp = 20
            xrange, yrange, zrange = np.zeros(samp), np.zeros(samp), np.zeros(samp)
            for k in range(0, samp):
                xrange[k] = self.rec[k][3][0]
                yrange[k] = self.rec[k][3][1]
                zrange[k] = self.rec[k][3][2]
            minimum_var = np.amin([np.var(xrange), np.var(yrange), np.var(zrange)])
            if minimum_var == np.var(xrange):
                delete_axis = "x"
            elif minimum_var == np.var(yrange):
                delete_axis = "y"
            elif minimum_var == np.var(zrange):
                delete_axis = "z"
            else:
                print("ERROR")
        for tim in tqdm(range(initial_time,int(initial_time+duration))):
            hist_accuml_partave = np.zeros(radius_lists.shape[0]-1)
            # make a group with the same particles
            for p in range(0, len(self.mlists)):
                if part_str1 in self.molcompose[p]:
                    mol1_count = self.molcount[p]
                if part_str2 in self.molcompose[p]:
                    mol2_count = self.molcount[p]
            new_part1 = np.zeros([int(mol1_count*part_count1), 4])
            new_part2 = np.zeros([int(mol2_count*part_count2), 4])
            count1, count2 = 0, 0
            for l in range(self.limrec[tim,0], self.limrec[tim,1]):
                if self.rec[l][0] == part_num1:
                    new_part1[count1,0] = self.rec[l][1]
                    new_part1[count1,1:4] = self.rec[l][3]
                    count1 += 1
                if self.rec[l][0] == part_num2:
                    new_part2[count2,0] = self.rec[l][1]
                    new_part2[count2, 1:4] = self.rec[l][3]
                    count2 += 1
            assert count1 == mol1_count*part_count1
            assert count2 == mol2_count*part_count2
            # measure the distance between them from one representative particle
            if self.dimension == 3:
                #write script for 3-D system
                for p in range(0, len(new_part1)):
                    hist_accuml = np.zeros(radius_lists.shape[0]-1)
                    another_part = np.zeros([new_part2.shape[0]*27, new_part2.shape[1]])
                    for x_ext in range(0, len(coor_list)):
                        for y_ext in range(0, len(coor_list)):
                            for z_ext in range(0, len(coor_list)):
                                c = int(x_ext + y_ext*3 + z_ext*9)
                                start = int(c*len(new_part2))
                                end = int((c+1)*len(new_part2))
                                another_part[start:end,:] = new_part2.copy()
                                another_part[start:end,1] += coor_list[x_ext]*self.xbox
                                another_part[start:end,2] += coor_list[y_ext]*self.xbox
                                another_part[start:end,3] += coor_list[z_ext]*self.zbox
                    copied_part = new_part1.copy()
                    copied_part[:,0] = 0 # DO NOT reset global number registeration
                    dist = another_part[:,:]-copied_part[p,:]
                    dist_wotgt = dist[np.any(np.array([dist[:,0]])!=new_part1[p,0], axis=0)]
                    distance = np.zeros([dist_wotgt.shape[0], 2])
                    distance[:,0] = dist_wotgt[:,0]
                    distance[:,1] = np.sqrt(dist_wotgt[:,1]**2+dist_wotgt[:,2]**2+dist_wotgt[:,3]**2)
                    dist_final = np.zeros(int(distance.shape[0]/27))
                    for d in range(0, len(dist_final)):
                        target = distance[np.any(np.array([distance[:,0]])==distance[d,0], axis=0)]
                        dist_final[d] = np.amin(target[:,1])
                    for r in range(0, len(radius_lists)-1):
                        minlim = radius_lists[r]
                        maxlim = radius_lists[r+1]
                        in_range = dist_final[(dist_final>minlim)&(dist_final<maxlim)]
                        hist_accuml[r] = len(in_range)
                    hist_accuml_partave += hist_accuml/len(new_part1)
            elif self.dimension == 2:
                # write scipt for 2-D system
                for p in range(0, len(new_part1)):
                    hist_accuml = np.zeros(radius_lists.shape[0]-1)
                    another_part = np.zeros([new_part2.shape[0]*27, new_part2.shape[1]])
                    for x_ext in range(0, len(coor_list)):
                        for y_ext in range(0, len(coor_list)):
                            for z_ext in range(0, len(coor_list)):
                                c = int(x_ext + y_ext*3 + z_ext*9)
                                start = int(c*len(new_part2))
                                end = int((c+1)*len(new_part2))
                                another_part[start:end,:] = new_part2.copy()
                                another_part[start:end,1] += coor_list[x_ext]*self.xbox
                                another_part[start:end,2] += coor_list[y_ext]*self.xbox
                                another_part[start:end,3] += coor_list[z_ext]*self.zbox
                    copied_part = new_part1.copy()
                    copied_part[:,0] = 0 # DO NOT reset global number registeration
                    dist = another_part[:,:]-copied_part[p,:]
                    dist_wotgt = dist[np.any(np.array([dist[:,0]])!=new_part1[p,0], axis=0)]
                    distance = np.zeros([dist_wotgt.shape[0], 2])
                    distance[:,0] = dist_wotgt[:,0]
                    if delete_axis == "x":
                        distance[:,1] = np.sqrt(dist_wotgt[:,2]**2+dist_wotgt[:,3]**2)
                    elif delete_axis == "y":
                        distance[:,1] = np.sqrt(dist_wotgt[:,1]**2+dist_wotgt[:,3]**2)
                    elif delete_axis == "z":
                        distance[:,1] = np.sqrt(dist_wotgt[:,1]**2+dist_wotgt[:,2]**2)
                    else:
                        print("ERROR")
                    dist_final = np.zeros(int(distance.shape[0]/27))
                    for d in range(0, len(dist_final)):
                        target = distance[np.any(np.array([distance[:,0]])==distance[d,0], axis=0)]
                        dist_final[d] = np.amin(target[:,1])
                    for r in range(0, len(radius_lists)-1):
                        minlim = radius_lists[r]
                        maxlim = radius_lists[r+1]
                        in_range = dist_final[(dist_final>minlim)&(dist_final<maxlim)]
                        hist_accuml[r] = len(in_range)
                    hist_accuml_partave += hist_accuml/len(new_part1)
            else:
                print("Dimension error: suspended analysis")
            hist_accuml_timeave += hist_accuml_partave/duration
        if self.dimension == 3:
            rho = len(new_part2)/(self.xbox*self.xbox*self.zbox)
            gr = hist_accuml_timeave/(4*rho*radius_slice*np.pi*(radius_lists[1:])**2)
        elif self.dimension == 2:
            if delete_axis == "z":
                rho = len(new_part2)/(self.xbox*self.xbox)
            else:
                rho = len(new_part2)/(self.xbox*self.zbox)
            gr = hist_accuml_timeave/(2*rho*radius_slice*np.pi*(radius_lists[1:]))
        else:
            pass
        return radius_lists[1:], gr


    def relative_distance_from_COM(self, part_str, part_count, tim, com, max_radius=100, radius_slice=0.1):
        radius_lists = np.arange(0, max_radius, radius_slice)
        coor_list = [-1, 0, 1]
        mol_size = self.size_list[self.types_str.index(part_str)]
        hist_accuml = np.zeros(radius_lists.shape[0]-1)
        hist_accuml_partave = np.zeros(radius_lists.shape[0]-1)
        # make a group with the same particles
        part_num = self.types_num[self.types_str.index(part_str)]
        for p in range(0, len(self.mlists)):
            if part_str in self.molcompose[p]:
                mol_count = self.molcount[p]
                break
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

