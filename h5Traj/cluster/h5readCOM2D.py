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

from .h5readCluster2D import h5readCluster2D

class h5readCOM2D( h5readCluster2D ): 
    def __init__(self, filename):
        super().__init__(filename)
        return None

    def detect_COM(self, cluster):
        cutoff = self.xbox/10
        leftlim = -self.xbox/2+cutoff
        rightlim = self.xbox/2-cutoff
        # check if the cluster position is beyond border
        # cluster : [globalnum, xpos, ypos, zpos, class_assign, types_num] * number of clusters
        resid_area = np.zeros(len(cluster))
        # residential area: [0, 1, 2, 3, 4, 10, 20, 30, 40]
        for c in range(0, len(cluster)):
            # examine the residential area of every particles
            area_num = 0
            for n in range(0,2):
                if cluster[c,int(n+1)] < leftlim:
                    num_x = 1
                elif cluster[c,int(n+1)] > rightlim:
                    num_x = 2
                else:
                    num_x = 0
                area_num += num_x*(10**n)
            resid_area[c] = area_num
        if len(np.unique(resid_area)) == 1:
            # calcuate Rg normally
            xcent = np.mean(cluster[:,1])
            ycent = np.mean(cluster[:,2])
            rg = np.sqrt(np.mean((cluster[:,1]-xcent)**2+(cluster[:,2]-ycent)**2))
            centers_info = np.array([xcent, ycent, rg])
        else:
            # set original residential area
            area = np.unique(resid_area)
            if np.any(area%10==1)==True and np.any(area%10==2)==True:
                # take a projection toward x axis
                hist, bins = np.histogram(cluster[:,1], bins=np.arange(-self.xbox/2, self.xbox/2, int(self.xbox/10)))
                if np.any(hist==0) == True:
                    x_border = bins[np.where(hist==0)[0][0]]
                else:
                    x_border= bins[np.where(hist==min(hist))[0][0]]
                cluster[np.where(cluster[:,1]>x_border), 1] -= self.xbox
            if np.any((area//10)%10==1)==True and np.any((area//10)%10==2)==True:
                # take a projection toward y axis
                hist, bins = np.histogram(cluster[:,2], bins=np.arange(-self.xbox/2, self.xbox/2, int(self.xbox/10)))
                if np.any(hist==0) == True:
                    y_border = bins[np.where(hist==0)[0][0]]
                else:
                    y_border = bins[np.where(hist==min(hist))[0][0]]
                cluster[np.where(cluster[:,2]>y_border), 2] -= self.xbox
            xcent = np.mean(cluster[:,1])
            ycent = np.mean(cluster[:,2])
            zcent = np.mean(cluster[:,3])
            rg = np.sqrt(np.mean((cluster[:,1]-xcent)**2+(cluster[:,2]-ycent)**2))
            centers_info = np.array([xcent, ycent, rg])
        return centers_info, cluster

    def detect_maximum(self, tim):
        molecules = super().whole_clustering(tim)
        cluster_lists = np.unique(molecules[:,4])
        largest_class = 0
        target = 1
        for c in range(0, len(cluster_lists)):
            classsize = len(molecules[np.any(np.array([molecules[:,4]])==cluster_lists[c], axis=0), :])
            if largest_class < classsize:
                largest_class = classsize
                target = c
        target_cluster = molecules[np.any(np.array([molecules[:,4]])==cluster_lists[target], axis=0), :]
        center_of_mass, target_rearranged = self.detect_COM(target_cluster)
        return center_of_mass, target_rearranged

    # WARNING: Now editing. Check the plot of COM if this program works properly.
    def detect_all(self, tim):
        molecules = super().whole_clustering(tim)
        cluster_lists = np.unique(molecules[:,4])
        cluster_list = []
        cluster_dict = {}
        cluster_com = {}
        for c in range(0, len(cluster_lists)):
            cluster_list.append(cluster_lists[c])
            target_cluster = molecules[np.any(np.array([molecules[:,4]])==cluster_lists[c],axis=0),:]
            center_of_mass, target_rearranged = self.detect_COM(target_cluster)
            cluster_dict[str(cluster_lists[c])] = target_rearranged
            cluster_com[str(cluster_lists[c])] = center_of_mass
        return cluster_list, cluster_dict, cluster_com

    def distance_distribution(self, tim):
        center_of_mass, target_rearranged = self.detect_maximum(tim)
        particle_variety = np.unique(target_rearranged[:,5])
        distrib = []
        plot_range = 2*self.xbox/3
        for p in particle_variety:
            specific_particles = target_rearranged[np.any(np.array([target_rearranged[:,5]])==p, axis=0),:] 
            distance = np.sqrt((specific_particles[:,1]-center_of_mass[0])**2 + (specific_particles[:,2]-center_of_mass[1])**2 + (specific_particles[:,3]-center_of_mass[2])**2)
            hist, bins = np.histogram(distance, bins=np.arange(0, plot_range, int(plot_range/25)))
            distrib.append([p, hist])
        return distrib, bins

    def plot_distribution(self, tim, specific=True):
        distrib, bins = self.distance_distribution(tim)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        for d in range(0, len(distrib)):
            numb = distrib[d][0]
            hist = distrib[d][1]
            if specific == True:
                if self.types_str[self.types_num.index(numb)] in ["A", "F", "G", "M"]:
                    ax.plot(bins[:-1], hist/np.sum(hist), label=self.types_str[self.types_num.index(numb)])
            else:
                ax.plot(bins[:-1], hist/np.sum(hist), label=self.types_str[self.types_num.index(numb)])
        ax.legend()
        plt.savefig("Distribution_from_COM2.png")
        plt.show()
        return None

    def smooth_distribution(self, start, duration, target_lists):
        # apply distribution to distance_distribution in each timestep
        fig = plt.figure()
        ax = fig.add_subplot(111)
        plot_range = 2*self.xbox/3
        distrib, bins = np.histogram(np.ones(25)*(-1), bins=np.arange(0, plot_range, int(plot_range/25)))
        distrib_smoothed = np.zeros([len(distrib), len(target_lists)])
        for tim in tqdm(range(start, start+duration)):
            distrib, bins = self.distance_distribution(tim)
            for d in range(0, len(distrib)):
                for tl in range(0, len(target_lists)):
                    if self.types_str[self.types_num.index(distrib[d][0])] == target_lists[tl]:
                        distrib_smoothed[:,tl] += distrib[d][1]
        #for d in range(0, len(distrib_smoothed)):
        #    for l in range(0, len(self.moltype)):
        #        if self.types_str[self.types_num.index(numb)] in itertools.chain.from_iterable(self.moltype[l]):
        for tl in range(0, len(target_lists)):
            hist = distrib_smoothed[:,tl]
            ax.plot(bins[:-1], hist/np.sum(hist), label=target_lists[tl])
        ax.legend()
        plt.savefig("Distribution_from_COM_smoothed2.png")
        plt.show()
        return None

    def xyz_projection(self, tim, projection="z"):
        center_of_mass, target_rearranged = self.detect_maximum(tim)
        particle_variety = np.unique(target_rearranged[:,5])
        distrib = []
        plot_range = self.xbox/2
        for p in particle_variety:
            specific_particles = target_rearranged[np.any(np.array([target_rearranged[:,5]])==p, axis=0),:] 
            if projection == "x":
                projected = specific_particles[:,1]-center_of_mass[0]
            elif projection == "y":
                projected = specific_particles[:,2]-center_of_mass[1]
            elif projection == "z":
                projected = specific_particles[:,3]-center_of_mass[2]
            else:
                print("INVALID PROJECTION DIRECTION: Please try again.")
                sys.exit()
            hist, bins = np.histogram(projected, bins=np.arange(-plot_range, plot_range, int(plot_range/25)))
            distrib.append([p, hist])
        return distrib, bins

    def xyz_slice(self, tim, projection="x"):
        center_of_mass, target_rearranged = self.detect_maximum(tim)
        particle_variety = np.unique(target_rearranged[:,5])
        distrib = []
        plot_range = self.xbox/2
        small_slice = 10
        for p in particle_variety:
            specific_particles = target_rearranged[np.any(np.array([target_rearranged[:,5]])==p, axis=0),:]
            if projection == "x":
                target_particles = specific_particles[np.abs(specific_particles[:,2]-center_of_mass[1])<small_slice,:]
                projected = target_particles[:,1]-center_of_mass[0]
            elif projection == "y":
                target_particles = specific_particles[np.abs(specific_particles[:,1]-center_of_mass[0])<small_slice,:]
                projected = target_particles[:,2]-center_of_mass[1]
            else:
                print("INVALID PROJECTION DIRECTION: Please try again.")
                sys.exit()
            hist, bins = np.histogram(projected, bins=np.arange(-plot_range, plot_range, int(plot_range/25)))
            distrib.append([p, hist])
        return distrib, bins

    def relative_xy_slice(self, tim, projection="x"):
        small_slice = 10
        cluster_list, cluster_dict, cluster_com = self.detect_all(tim)
        distrib = {}
        for current in cluster_list:
            center_of_mass = cluster_com[str(current)]
            target_rearranged = cluster_dict[str(current)]
            if len(target_rearranged) > 200: # Extract only large cluster with more than 100 particles
                radius_of_gyration = center_of_mass[2]
                particle_variety = np.unique(target_rearranged[:,5])
                for p in particle_variety:
                    specific_particles = target_rearranged[np.any(np.array([target_rearranged[:,5]])==p, axis=0),:] 
                    if projection == "x":
                        target_particles = specific_particles[np.abs(specific_particles[:,2]-center_of_mass[1])<small_slice,:] 
                        projected = target_particles[:,1]-center_of_mass[0]
                    elif projection == "y":
                        target_particles = specific_particles[np.abs(specific_particles[:,1]-center_of_mass[0])<small_slice,:]
                        projected = target_particles[:,2]-center_of_mass[1]
                    else:
                        print("INVALID PROJECTION DIRECTION: Please try again.")
                        sys.exit()
                    hist, bins = np.histogram(projected/radius_of_gyration, bins=np.arange(-2.0, 2.0, 2.0*2/50))
                    #print(p, hist)
                    if str(p) in distrib:
                        distrib[str(p)] += hist
                    else:
                        distrib[str(p)] = hist
        return distrib, bins

    def plot_projection(self, tim, specific=True, projection="x"):
        distrib, bins = self.xyz_projection(tim, "x")
        fig = plt.figure()
        ax = fig.add_subplot(111)
        for d in range(0, len(distrib)):
            numb = distrib[d][0]
            hist = distrib[d][1]
            if specific == True:
                if self.types_str[self.types_num.index(numb)] in ["A", "F", "G", "M"]:
                    ax.plot(bins[:-1], hist/np.sum(hist), label=self.types_str[self.types_num.index(numb)])
            else:
                ax.plot(bins[:-1], hist/np.sum(hist), label=self.types_str[self.types_num.index(numb)])
        ax.legend()
        plt.savefig("projection_to_Z.png")
        plt.show()
        return None

    def smooth_projection(self, start, duration, target_lists, color_lists, pngname, projection="z"):
        # apply distribution to distance_distribution in each timestep
        fig = plt.figure()
        ax = fig.add_subplot(111)
        plot_range = self.xbox/2
        distrib, bins = np.histogram(np.ones(25)*(-1), bins=np.arange(-plot_range, plot_range, int(plot_range/25)))
        distrib_smoothed = np.zeros([len(distrib), len(target_lists)])
        for tim in tqdm(range(start, start+duration)):
            distrib, bins = self.xyz_projection(tim, projection)
            for d in range(0, len(distrib)):
                for tl in range(0, len(target_lists)):
                    if self.types_str[self.types_num.index(distrib[d][0])] == target_lists[tl]:
                        distrib_smoothed[:,tl] += distrib[d][1]
        for tl in range(0, len(target_lists)):
            hist = distrib_smoothed[:,tl]
            ax.plot(bins[:-1], hist/np.sum(hist), label=target_lists[tl], color=color_lists[tl])
        ax.legend()
        plt.savefig(pngname)
        plt.show()
        return None

    def plot_slice(self, tim, specific=True, projection="z"):
        distrib, bins = self.xyz_slice(tim, projection)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        for d in range(0, len(distrib)):
            numb = distrib[d][0]
            hist = distrib[d][1]
            if specific == True:
                if self.types_str[self.types_num.index(numb)] in ["A", "F", "G", "M"]:
                    ax.plot(bins[:-1], hist/np.sum(hist), label=self.types_str[self.types_num.index(numb)])
            else:
                ax.plot(bins[:-1], hist/np.sum(hist), label=self.types_str[self.types_num.index(numb)])
        ax.legend()
        plt.savefig("slice_of_Z3.png")
        plt.show()
        return None

    def smooth_slice(self, start, duration, target_lists, color_lists, pngname, projection="y"):
        # apply distribution to distance_distribution in each timestep
        fig = plt.figure()
        ax = fig.add_subplot(111)
        plot_range = self.xbox/2
        distrib, bins = np.histogram(np.ones(25)*(-1), bins=np.arange(-plot_range, plot_range, int(plot_range/25)))
        distrib_smoothed = np.zeros([len(distrib), len(target_lists)])
        for tim in tqdm(range(start, start+duration)):
            distrib, bins = self.xyz_slice(tim, projection)
            for d in range(0, len(distrib)):
                for tl in range(0, len(target_lists)):
                    if self.types_str[self.types_num.index(distrib[d][0])] == target_lists[tl]:
                        distrib_smoothed[:,tl] += distrib[d][1]
        for tl in range(0, len(target_lists)):
            hist = distrib_smoothed[:,tl]
            ax.plot(bins[:-1], hist/np.sum(hist), label=target_lists[tl], color=color_lists[tl])
            newarray = np.vstack([[bins[:-1]], [hist/np.sum(hist)]])
            np.savetxt("data"+pngname+target_lists[tl]+".dat", newarray.T)
            #np.savetxt("distrib"+pngname+target_lists[tl]+".dat", hist)
        ax.legend()
        plt.savefig(pngname)
        plt.show()
        return None

    def smooth_relative_slice(self, start, duration, target_lists, color_lists, pngname, projection="x"):
        # apply distribution to distance_distribution in each timestep
        fig = plt.figure()
        ax = fig.add_subplot(111)
        plot_range = 2.0
        distrib, bins = np.histogram(np.ones(25)*(-1), bins=np.arange(-plot_range, plot_range, plot_range*2/50))
        distrib_smoothed = np.zeros([len(distrib), len(target_lists)])
        for tim in tqdm(range(start, start+duration)):
            distrib, bins = self.relative_xy_slice(tim, projection)
            #print(distrib["0.0"])
            ### write again from distrib(list) to distrib(dict) ###
            keys = list(distrib.keys())
            distrib_keys = [float(s) for s in keys]
            #print(distrib_keys)
            for d in range(0, len(distrib)):
                for tl in range(0, len(target_lists)):
                    #print(distrib_keys[d])
                    #print(int(distrib_keys[d]))
                    if self.types_str[self.types_num.index(int(distrib_keys[d]))] == target_lists[tl]:
                        distrib_smoothed[:,tl] += distrib[str(distrib_keys[d])] ## problem here
                        #print(distrib_smoothed[:,tl])
        for tl in range(0, len(target_lists)):
            hist = distrib_smoothed[:,tl]
            ax.plot(bins[:-1], hist/np.sum(hist), label=target_lists[tl], color=color_lists[tl])
            newarray = np.vstack([[bins[:-1]], [hist/np.sum(hist)]])
            np.savetxt("data"+pngname+target_lists[tl]+".dat", newarray.T)
        ax.legend()
        plt.savefig(pngname)
        return None

    def draw_snapshot(self, tim, mol="whole", mol_str="all", part_in_mol=1):
        if mol != "whole":
            oldcl = super().particle_clustering(tim, mol, mol_str, part_in_mol)
            #cents, largest, largest_rg = self.true_particle_clustering(tim, mol, mol_str, part_in_mol)
        else:
            oldcl = super().whole_clustering(tim)
            #cents, largest, largest_rg = self.true_whole_clustering(tim)
        #print(cents[:,:], largest, largest_rg)
        COM, newcl = self.detect_maximum(tim) 
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.set_xlim(-self.xbox/2, self.xbox/2)
        ax.set_ylim(-self.xbox/2, self.xbox/2)
        ax.set_zlim(-self.zbox/2, self.zbox/2)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.set_aspect("equal")
        ax.scatter(newcl[:,1], newcl[:,2], newcl[:,3], s=20)
        #for c in range(0, len(cents[:,0])):
        #    ax.add_patch(patches.Circle(xy=(cents[c,0], cents[c,1]), radius=cents[c,2], fc='orange', ec='black', alpha=0.2))
        #    ax.text(cents[c,0], cents[c,1], str(int(cents[c,3])), size=20, horizontalalignment="center", verticalalignment="center")
        plt.savefig("clustering3_"+mol_str+"in"+mol+"at"+str(tim)+".png")
        plt.show()
        return None

