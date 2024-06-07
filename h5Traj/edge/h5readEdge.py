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

# classify the way of bonding as below:
#
#       AMPAR bonding varieties
#    0       1           2          3       4
#    x   |   x   |   o      x   |   o   |   o   
#  x-o-x | o-o-x | o-o-x  o-o-o | o-o-o | o-o-o 
#    x   |   x   |   x      x   |   x   |   o   
#
#       PSD-95 bonding varieties
#    0              1                     2              3
#  x-x-x | o-x-x  x-o-x  x-x-o | o-o-x  o-x-o  x-o-o | o-o-o
# 

from .. import h5readTraj

class h5readEdge( h5readTraj ):
    def __init__(self, filename, dim):
        super().__init__(filename, dim)
        # count nonzero value of n_molecule
        self.edges = self.topo['edges']
        self.limedges = self.topo['limitsEdges'] 
        self.variables = []
        return None
   
    def connected_edge(self, tim):
        timedges = self.edges[self.limedges[tim,0]:self.limedges[tim,1]]
        timparts = self.parts[self.limparts[tim,0]:self.limparts[tim,1]]
        index = 0
        # first, change local edgelists into global edgelists
        local_partlist = []
        global_edgelist = np.empty((0,2))
        while index < (self.limparts[tim,1]-self.limparts[tim,0]):
            topvalue = timparts[index]
            local_partlist.append(np.array(timparts[int(index+1):int(index+topvalue+1)]))
            index = int(index+topvalue+1)
        index2 = 0
        i = 0
        while index2 < (self.limedges[tim,1]-self.limedges[tim,0]):
            edgevalue = timedges[index2,0]
            local_edge = timedges[int(index2+1):int(index2+edgevalue+1)]
            global_part = local_partlist[i][:]
            lfunc = lambda x: global_part[x]
            global_part = local_partlist[i][:]
            global_edge = lfunc(local_edge)
            global_edgelist = np.concatenate([global_edgelist, global_edge])
            index2 = int(index2+edgevalue+1)
            i += 1
        return global_edgelist

    def global_numberlists(self):
        molecule_global = {} # dict, example: "A1": [lists]
        for j in range(self.limrec[0,0], self.limrec[0,1]):
            temp_mol = self.types_str[self.types_num.index(int(self.rec[j][0]))] # search molecule type
            if temp_mol not in molecule_global:
                molecule_global[temp_mol] =[]
            molecule_global[temp_mol].append(self.rec[j][1]) # add global number 
        return molecule_global

    #def check_multiplicity(fname, tim, a_part, p_part):
    #    edge_initial = connected_edge(0)
    #    edge_now = connected_edge(tim)
    #    global_numberlists = global_numberlists()
    #    ### from here, we should detect the type of the particles
    #    topology_assign = np.zeros(a_part*5+p_part*6)
    #    #print(len(topology_assign))
    #    topological_edges = np.zeros(edge_now.shape)
    #    for a in range(0, len(ampar_global)):
    #        ampar_complex = np.unique(np.ravel(edge_now[np.any(edge_now==ampar_global[a], axis=1)]))
    #        for comp in ampar_complex:
    #            topology_assign[int(comp)] = a
    #    for p in range(0, len(sh3_global)):
    #        pdz3_candidate = np.unique(np.ravel(edge_now[np.any(edge_now==sh3_global[p], axis=1)]))
    #        # from candidates of pdz3, find pdz3 particles from pdz_global
    #        for cand in pdz3_candidate:
    #            topology_assign[int(cand)] = p + 1000
    #            if cand in pdz_global:
    #                #print(cand)
    #                pdz2_candidate = np.unique(np.ravel(edge_now[np.any(edge_now==cand, axis=1)]))
    #                #print("p2cand: ", pdz2_candidate)
    #        for cand2 in pdz2_candidate:
    #            if cand2 in pdz_global and cand2 not in pdz3_candidate:
    #                #print(cand2)
    #                topology_assign[int(cand2)] = p+1000
    #                pdz1_candidate = np.unique(np.ravel(edge_now[np.any(edge_now==cand2, axis=1)]))
    #                #print("p1cand: ", pdz1_candidate)
    #        for cand3 in pdz1_candidate:
    #            if cand3 in pdz_global and cand3 not in pdz2_candidate:
    #                #print(cand3)
    #                topology_assign[int(cand3)] = p+1000
    #                nterm = list(set(np.unique(np.ravel(edge_now[np.any(edge_now==cand3, axis=1)]))) & set(nterm_global))
    #                topology_assign[int(nterm[0])] = p+1000
    #    assert len(topology_assign[topology_assign<1000]) == a_part*5
    #    #print("passed")
    #    tpinteraction = []
    #    for e in range(0, len(edge_now)):
    #        topological_edges[e,:] = np.array([topology_assign[int(edge_now[e,0])], topology_assign[int(edge_now[e,1])]])
    #        if topology_assign[int(edge_now[e,0])] != topology_assign[int(edge_now[e,1])]:
    #            tpinteraction.append(topological_edges[e,:])
    #    #print(tpinteraction)
    #    tp, tpcount = np.unique(tpinteraction, axis=0, return_counts=True)
    #    #print(tpcount)
    #    ######## drawing graphi
    #    #g1 = nx.Graph()
    #    #g1.add_nodes_from(np.arange(0,int(a_part)))
    #    #g1.add_nodes_from(np.arange(1000,1000+int(p_part)))
    #    #print(topology_client)
    #    #print(count)
    #    #color_map = []
    #    #for node in g1:
    #    #    if node < a_part:
    #    #        color_map.append('red')
    #    #    else: color_map.append('blue')  
    #    #for tp in range(0, topology_client.shape[0]):
    #    #    g1.add_edge(topology_client[tp,0], topology_client[tp,1], alpha=0.2)
    #    #nx.draw(g1, with_labels = True, node_color=color_map)
    #    #plt.savefig("r200200_multiple.png")
    #    #plt.show()
    #    return tpcount

#check_multiplicity("v2p424.264phase200_7.h5", 10000, 200, 200)
#sys.exit()
#dur = 50
#iteration = 5
#x = np.array([1, 2, 3])
#slaba = np.zeros([iteration, 3])
#memba = np.zeros([iteration, 3])
#multibins = [0.5, 1.5, 2.5, 3.5]
#for k in range(0, iteration):
#    slab_snap = np.zeros([dur, 3])
#    memb_snap = np.zeros([dur, 3])
#    for t in tqdm(range(0, dur)):
#        slab_snap[t,:], sbins = np.histogram(check_multiplicity("v2p212.132phase50_3_original.h5", t+int(4000+k*dur), 50, 50), bins=multibins)
#        memb_snap[t,:], mbins = np.histogram(check_multiplicity("v2p212.132phase50_strict.h5", t+int(9000+k*dur), 50, 50), bins=multibins)
#    slaba[k,:] = np.mean(slab_snap, axis=0)
#    memba[k,:] = np.mean(memb_snap, axis=0)
#slab = np.mean(slaba, axis=0)
#memb = np.mean(memba, axis=0)
#slabs = np.std(slaba, axis=0)/np.sqrt(iteration)
#membs = np.std(memba, axis=0)/np.sqrt(iteration)
##weight1 = np.ones(len(slab))/float(len(slab))
##weight2 = np.ones(len(memb))/float(len(memb))
#fig = plt.figure(figsize=[3,3])
#margin = 0.2
##slab = np.mean(slab_snap, axis=0)
##memb = np.mean(memb_snap, axis=0)
#print(slab, memb)
##plt.hist([slab, memb], bins=[0.5, 1.5, 2.5, 3.5], stacked=False, label=["3D", "2D"], color=[palette[4], palette[1]])
#plt.bar(x+margin, memb, color="coral", yerr=slabs, width=margin*2, label="strict")
#plt.bar(x-margin, slab, color=palette[4], yerr=membs, width=margin*2, label="original")
#plt.xticks([1,2,3])
##plt.ylim(0,550)
#plt.ylabel("Averaged number of molecules", fontsize=11)
#plt.xlabel("Multiplicity", fontsize=11)
#plt.legend(framealpha=0)
#plt.savefig("multiplicity_compare.svg")
#plt.show()


