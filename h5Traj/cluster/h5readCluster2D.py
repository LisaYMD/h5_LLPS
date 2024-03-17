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

from .h5readCluster import h5readCluster

class h5readCluster2D( h5readCluster ): 
    def __init__(self, filename):
        super().__init__(filename, 2)
        return None
 
    # input the direction of correction (depends on target residential ID)
    def beyond_boundary(self, cutoff, ampar, resid_id, t_id, i_list, x_direct, y_direct):
        target = resid_id[t_id][1:]
        for t in target:
            for i in range(0, len(i_list)):
                if len(resid_id[i]) != 1:
                    # try comparison between target to candidate
                    for c in resid_id[i][1:]:
                        numt = len(ampar[np.any(np.array([ampar[:,4]])==t, axis=0)])
                        numc = len(ampar[np.any(np.array([ampar[:,4]])==c, axis=0)])
                        for p1, p2 in itertools.product(range(0,numt), range(0,numc)):
                            t1 = ampar[np.any(np.array([ampar[:,4]])==t, axis=0)][p1,:]
                            c1 = ampar[np.any(np.array([ampar[:,4]])==c, axis=0)][p2,:]
                            dist = np.sqrt((t1[1]-(c1[1]+x_direct*self.xbox))**2+(t1[2]-(c1[2]+y_direct*self.xbox))**2)
                            if dist < cutoff:
                                # make it as a same cluster and break
                                ampar[np.any(np.array([ampar[:,4]])==c, axis=0), 4] = t
                                resid_id[i][1:].remove(c)
                                break
        return ampar, resid_id

    def radius_gyration(self, cluster, isFinal=False):
        cutoff = self.xbox/10
        leftlim = -self.xbox/2+cutoff
        rightlim = self.xbox/2-cutoff
        # check if the cluster position is beyond border
        # cluster : [globalnum, xpos, ypos, class_assign] * number of clusters
        resid_area = np.zeros(len(cluster))
        for c in range(0, len(cluster)):
            # examine the residential area of every particles
            if cluster[c,1] < leftlim:
                if cluster[c,2] < leftlim:
                    resid_area[c] = 13
                elif cluster[c,2] > rightlim:
                    resid_area[c] = 14
                else:
                    resid_area[c] = 10
            elif cluster[c,1] > rightlim:
                if cluster[c,2] < leftlim:
                    resid_area[c] = 23
                elif cluster[c,2] > rightlim:
                    resid_area[c] = 24
                else:
                    resid_area[c] = 20
            elif cluster[c,2] < leftlim:
                resid_area[c] = 3
            elif cluster[c,2] > rightlim:
                resid_area[c] = 4
            else:
                resid_area[c] = 0
        if len(np.unique(resid_area)) == 1:
            # calcuate Rg normally
            xcent = np.mean(cluster[:,1])
            ycent = np.mean(cluster[:,2])
            rg = np.sqrt(np.mean((cluster[:,1]-xcent)**2+(cluster[:,2]-ycent)**2))
            centers_info = np.array([xcent, ycent, rg, len(cluster)])
        else:
            # set original residential area
            area = np.unique(resid_area)
            if np.any((area>=10)&(area<20))==True and np.any(area>=20)==True:
                # take a projection toward x axis
                hist, bins = np.histogram(cluster[:,1], bins=np.arange(-self.xbox/2, self.xbox/2, int(self.xbox/10)))
                if np.any(hist==0) == True:
                    x_border = bins[np.where(hist==0)[0][0]]
                else:
                    x_border= bins[np.where(hist==min(hist))[0][0]]
                cluster[np.where(cluster[:,1]>x_border), 1] -= self.xbox
            if np.any(area%10==3)==True and np.any(area%10==4)==True:
                # take a projection toward y axis
                hist, bins = np.histogram(cluster[:,2], bins=np.arange(-self.xbox/2, self.xbox/2, int(self.xbox/10)))
                if np.any(hist==0) == True:
                    y_border = bins[np.where(hist==0)[0][0]]
                else:
                    y_border = bins[np.where(hist==min(hist))[0][0]]
                cluster[np.where(cluster[:,2]>y_border), 2] -= self.xbox
            xcent = np.mean(cluster[:,1])
            ycent = np.mean(cluster[:,2])
            rg = np.sqrt(np.mean((cluster[:,1]-xcent)**2+(cluster[:,2]-ycent)**2))
            if isFinal == True:
                if xcent < -self.xbox/2:
                    xcent += self.xbox
                if ycent < -self.xbox/2:
                    ycent += self.xbox
            centers_info = np.array([xcent, ycent, rg, len(cluster)])
        return centers_info

    def true_particle_clustering(self, tim, mol, mol_str, part_in_mol):
        ampar = super().particle_clustering(tim, mol, mol_str, part_in_mol)
        # first, count the number of class (old)
        cutoff = self.size_list[self.types_str.index(mol_str)]*2 + 1.5 # this is new cutoff radius
        isSeparate = False
        while isSeparate == False:
            classold = np.unique(ampar[:,4])
            for k, l in itertools.combinations(classold, 2):
            # ampar[global number, x position, y position, z position, class assignment]
                numk = len(ampar[np.any(np.array([ampar[:,4]])==k, axis=0)])
                numl = len(ampar[np.any(np.array([ampar[:,4]])==l, axis=0)])
                #print(numk, numl)
                for p1, p2 in itertools.product(range(0,numk), range(0,numl)):
                    k1 = ampar[np.any(np.array([ampar[:,4]])==k, axis=0)][p1,:]
                    l1 = ampar[np.any(np.array([ampar[:,4]])==l, axis=0)][p2,:]
                    dist = np.sqrt((k1[1]-l1[1])**2+(k1[2]-l1[2])**2+(k1[3]-l1[3])**2)
                    if dist < cutoff:
                        #make it as a same cluster and break
                        ampar[np.any(np.array([ampar[:,4]])==l, axis=0), 4] = k
                        break
            if len(classold) == len(np.unique(ampar[:,4])):
                isSeparate = True
        # classify as a new class
        classmid = np.unique(ampar[:,4])
        # judge whether the class has a member in area A to D
        leftlim = -self.xbox/2+cutoff
        rightlim = self.xbox/2-cutoff
        resid_id = [[0,], [1,], [2,], [3,], [4,], [13,], [14,], [23,], [24,]]
        for cm in range(0, len(classmid)):
            # in each class, separate the residential area
            # residential classification: [0, 1, 2, 3, 4, 13, 14, 23, 24]
            member_x = ampar[np.any(np.array([ampar[:,4]])==classmid[cm], axis=0), 1]
            member_y = ampar[np.any(np.array([ampar[:,4]])==classmid[cm], axis=0), 2]
            # making a list: [residential id,[classno1, classno2, ...]]
            if np.any( member_x < leftlim ) == True:
                if np.any( member_y < leftlim ) == True:
                    # in area AC (13)
                    resid_id[5].append(classmid[cm])
                elif np.any( member_y > rightlim ) == True:
                    # in area AD (14)
                    resid_id[6].append(classmid[cm])
                else:
                    # in area A (1)
                    resid_id[1].append(classmid[cm])
            elif np.any( member_x > rightlim ) == True:
                if np.any( member_y < leftlim ) == True:
                    # in area BC (23)
                    resid_id[7].append(classmid[cm])
                elif np.any( member_y > rightlim ) == True:
                    # in area BD (24)
                    resid_id[8].append(classmid[cm])
                else:
                    # in area B (2)
                    resid_id[2].append(classmid[cm])
            elif np.any( member_y < leftlim ) == True:
                # in area C (3)
                resid_id[3].append(classmid[cm])
            elif np.any( member_y > rightlim ) == True:
                # in area D (4)
                resid_id[4].append(classmid[cm])
            else:
                # in area O (0)
                resid_id[0].append(classmid[cm])
        # treat class other than that with residential number with 0 as "special" in calculation of Rg
        isBeyondBoundary = False
        while isBeyondBoundary == False:
            classquad = np.unique(ampar[:,4])
            # resid 1 -> resid 2, 23, 24
            if len(resid_id[1]) != 1:
                ampar, resid_id = self.beyond_boundary(cutoff, ampar, resid_id, 1, [2, 7, 8], -1, 0)
            # resid 2 -> resid 1, 13, 14
            if len(resid_id[2]) != 1:
                ampar, resid_id = self.beyond_boundary(cutoff, ampar, resid_id, 2, [1, 5, 6], 1, 0)
            # resid 3 -> resid 4, 14, 24
            if len(resid_id[3]) != 1:
                ampar, resid_id = self.beyond_boundary(cutoff, ampar, resid_id, 3, [4, 6, 8], 0, -1)
            # resid 4 -> resid 3, 13, 23
            if len(resid_id[4]) != 1:
                ampar, resid_id = self.beyond_boundary(cutoff, ampar, resid_id, 4, [3, 5, 7], 0, 1)
            # resid 13 -> resid 14, 23, 24
            if len(resid_id[5]) != 1:
                ampar, resid_id = self.beyond_boundary(cutoff, ampar, resid_id, 5, [6, 7, 8], -1, -1)
            # resid 14 -> resid 23, 24
            if len(resid_id[6]) != 1:
                ampar, resid_id = self.beyond_boundary(cutoff, ampar, resid_id, 6, [7, 8], -1, 1)
            # resid 23 -> resid 24
            if len(resid_id[7]) != 1:
                ampar, resid_id = self.beyond_boundary(cutoff, ampar, resid_id, 7, [8], 1, -1)
            if len(classquad) == len(np.unique(ampar[:,4])):
                isBeyondBoundary = True
        # classify as a new class
        classnew = np.unique(ampar[:,4])
        centers_new = np.zeros([len(classnew), 4]) # records the position of center and radius of gyration
        # choose the class, rename it and calculate the radius of gyration
        for cn in range(0, len(classnew)):
            ampar[np.any(np.array([ampar[:,4]])==classnew[cn], axis=0),4] = cn
            whole2 = ampar[np.any(np.array([ampar[:,4]])==cn, axis=0)]
            centers_new[int(cn-1), :] = self.radius_gyration(whole2, cutoff)
        # find the largest cluster
        largest_rg = centers_new[np.any(np.array([centers_new[:,3]])==np.max(centers_new[:,3]), axis=0), 2]
        largest = centers_new[np.any(np.array([centers_new[:,3]])==np.max(centers_new[:,3]), axis=0),3]
        return centers_new, largest, largest_rg

    def true_whole_clustering(self, tim):
        start = time.time()
        ampar = super().whole_clustering(tim) # output: [global number, xpos, ypos, zpos, class, types_num]
        # first, count the number of class (old)
        cutoff = self.size_list[self.size_list==max(self.size_list)]*2 +1.5
        isSeparate = False
        checkpoint1 = time.time()
        print(checkpoint1-start)
        while isSeparate == False:
            classold = np.unique(ampar[:,4])
            for k, l in itertools.combinations(classold, 2):
                # ampar[global number, x position, y position, z position, class assignment]
                numk = len(ampar[np.any(np.array([ampar[:,4]])==k, axis=0)])
                numl = len(ampar[np.any(np.array([ampar[:,4]])==l, axis=0)])
                for p1, p2 in itertools.product(range(0,numk), range(0,numl)):
                    k1 = ampar[np.any(np.array([ampar[:,4]])==k, axis=0)][p1,:]
                    l1 = ampar[np.any(np.array([ampar[:,4]])==l, axis=0)][p2,:]
                    dist = np.sqrt((k1[1]-l1[1])**2+(k1[2]-l1[2])**2+(k1[3]-l1[3])**2)
                    if dist < cutoff:
                        #make it as a same cluster and break
                        ampar[np.any(np.array([ampar[:,4]])==l, axis=0), 4] = k
                        break
            if len(classold) == len(np.unique(ampar[:,4])):
                isSeparate = True
        # classify as a new class
        classmid = np.unique(ampar[:,4])
        checkpoint2 = time.time()
        print(checkpoint2-start)
        print(classmid)
        # judge whether the class has a member in area A to D
        leftlim = -self.xbox/2+cutoff
        rightlim = self.xbox/2-cutoff
        resid_id = [[0,], [1,], [2,], [3,], [4,], [13,], [14,], [23,], [24,]]
        for cm in range(0, len(classmid)):
            # in each class, separate the residential area
            # residential classification: [0, 1, 2, 3, 4, 13, 14, 23, 24]
            member_x = ampar[np.any(np.array([ampar[:,4]])==classmid[cm], axis=0), 1]
            member_y = ampar[np.any(np.array([ampar[:,4]])==classmid[cm], axis=0), 2]
            # making a list: [residential id,[classno1, classno2, ...]]
            if np.any( member_x < leftlim ) == True:
                if np.any( member_y < leftlim ) == True:
                    # in area AC (13)
                    resid_id[5].append(classmid[cm])
                elif np.any( member_y > rightlim ) == True:
                    # in area AD (14)
                    resid_id[6].append(classmid[cm])
                else:
                    # in area A (1)
                    resid_id[1].append(classmid[cm])
            elif np.any( member_x > rightlim ) == True:
                if np.any( member_y < leftlim ) == True:
                    # in area BC (23)
                    resid_id[7].append(classmid[cm])
                elif np.any( member_y > rightlim ) == True:
                    # in area BD (24)
                    resid_id[8].append(classmid[cm])
                else:
                    # in area B (2)
                    resid_id[2].append(classmid[cm])
            elif np.any( member_y < leftlim ) == True:
                # in area C (3)
                resid_id[3].append(classmid[cm])
            elif np.any( member_y > rightlim ) == True:
                # in area D (4)
                resid_id[4].append(classmid[cm])
            else:
                # in area O (0)
                resid_id[0].append(classmid[cm])
        # treat class other than that with residential number with 0 as "special" in calculation of Rg
        isBeyondBoundary = False
        while isBeyondBoundary == False:
            classquad = np.unique(ampar[:,4])
            # resid 1 -> resid 2, 23, 24
            if len(resid_id[1]) != 1:
                ampar, resid_id = self.beyond_boundary(cutoff, ampar, resid_id, 1, [2, 7, 8], -1, 0)
            # resid 2 -> resid 1, 13, 14
            if len(resid_id[2]) != 1:
                ampar, resid_id = self.beyond_boundary(cutoff, ampar, resid_id, 2, [1, 5, 6], 1, 0)
            # resid 3 -> resid 4, 14, 24
            if len(resid_id[3]) != 1:
                ampar, resid_id = self.beyond_boundary(cutoff, ampar, resid_id, 3, [4, 6, 8], 0, -1)
            # resid 4 -> resid 3, 13, 23
            if len(resid_id[4]) != 1:
                ampar, resid_id = self.beyond_boundary(cutoff, ampar, resid_id, 4, [3, 5, 7], 0, 1)
            # resid 13 -> resid 14, 23, 24
            if len(resid_id[5]) != 1:
                ampar, resid_id = self.beyond_boundary(cutoff, ampar, resid_id, 5, [6, 7, 8], -1, -1)
            # resid 14 -> resid 23, 24
            if len(resid_id[6]) != 1:
                ampar, resid_id = self.beyond_boundary(cutoff, ampar, resid_id, 6, [7, 8], -1, 1)
            # resid 23 -> resid 24
            if len(resid_id[7]) != 1:
                ampar, resid_id = self.beyond_boundary(cutoff, ampar, resid_id, 7, [8], 1, -1)
            if len(classquad) == len(np.unique(ampar[:,4])):
                isBeyondBoundary = True
        # classify as a new class
        classnew = np.unique(ampar[:,4])
        centers_new = np.zeros([len(classnew), 4]) # records the position of center and radius of gyration
        # choose the class, rename it and calculate the radius of gyration
        for cn in range(0, len(classnew)):
            ampar[np.any(np.array([ampar[:,4]])==classnew[cn], axis=0),4] = cn
            whole2 = ampar[np.any(np.array([ampar[:,4]])==cn, axis=0)]
            centers_new[int(cn-1), :] = self.radius_gyration(whole2, True)
        # find the largest cluster
        largest_rg = centers_new[np.any(np.array([centers_new[:,3]])==np.max(centers_new[:,3]), axis=0), 2]
        largest = centers_new[np.any(np.array([centers_new[:,3]])==np.max(centers_new[:,3]), axis=0),3]
        return centers_new, largest, largest_rg

    def draw_snapshot(self, tim, mol="whole", mol_str="all", part_in_mol=1):
        if mol != "whole":
            oldcl = super().particle_clustering(tim, mol, mol_str, part_in_mol)
            cents, largest, largest_rg = self.true_particle_clustering(tim, mol, mol_str, part_in_mol)
        else:
            oldcl = super().whole_clustering(tim)
            cents, largest, largest_rg = self.true_whole_clustering(tim)
        #print(cents[:,:], largest, largest_rg)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_xlim(-self.xbox/2, self.xbox/2)
        ax.set_ylim(-self.xbox/2, self.xbox/2)
        ax.set_aspect("equal")
        ax.scatter(oldcl[:,1], oldcl[:,2], s=20)
        for c in range(0, len(cents[:,0])):
            ax.add_patch(patches.Circle(xy=(cents[c,0], cents[c,1]), radius=cents[c,2], fc='orange', ec='black', alpha=0.2))
            ax.text(cents[c,0], cents[c,1], str(int(cents[c,3])), size=20, horizontalalignment="center", verticalalignment="center")
        plt.savefig("clustering_"+mol_str+"in"+mol+"at"+str(tim)+".png")
        plt.show()
        return None

    def output_largestcluster(self, outname, duration):
        largest_trans = np.zeros([duration,3])
        for t in tqdm(range(0, duration)):
            cents, largest, largest_rg = self.true_whole_clustering(t)
            largest_trans[t,0] = t
            largest_trans[t,1] = largest[0]
            largest_trans[t,2] = largest_rg[0]
        np.savetxt(outname, largest_trans)
        return None

    def output_particle_largestcluster(self, outname, duration, mol, mol_str, part_in_mol):
        largest_trans = np.zeros([duration,3])
        for t in tqdm(range(0, duration)):
            cents, largest, largest_rg = self.true_particle_clustering(t, mol, mol_str, part_in_mol)
            largest_trans[t,0] = t
            largest_trans[t,1] = largest[0]
            largest_trans[t,2] = largest_rg[0]
        np.savetxt(outname, largest_trans)
        return None

