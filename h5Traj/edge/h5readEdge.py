import sys
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import h5py
import hdf5plugin
import itertools
from tqdm import tqdm
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

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
np.set_printoptions(threshold=300)

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

    def global_numberlists(self, tim):
        molecule_global = {} # dict, example: "A1": [lists]
        for j in range(self.limrec[tim,0], self.limrec[tim,1]):
            temp_mol = self.types_str[self.types_num.index(int(self.rec[j][0]))] # search molecule type
            if temp_mol not in molecule_global:
                molecule_global[temp_mol] = []
            molecule_global[temp_mol].append(self.rec[j][1]) # add global number 
        return molecule_global

    def particle_adjacent_matrix(self, edges):
        most = max(np.ravel(edges))
        adj = np.zeros([int(most+1), int(most+1)])
        for e in range(0, len(edges)):
            adj[int(edges[e,0]), int(edges[e,1])] = 1
            adj[int(edges[e,1]), int(edges[e,0])] = 1
        assert np.array_equal(adj, adj.T)
        return adj

    def plot_part_adj(self, edges):
        adj = self.particle_adjacent_matrix(edges)
        plt.imshow(adj)
        plt.show()
        return None

    def adj_to_edges(self, adj):
        edgelist = []
        for i in range(0, len(adj)):
            for j in range(0, len(adj)):
                if adj[i,j] == 1:
                    edgelist.append(np.array([i,j]))
        edgelist = np.array(edgelist)
        return edgelist

    # assign global numbers to each molecules
    def global_protlists(self):
        edge_initial = self.connected_edge(0)
        adjacent_matrix = self.particle_adjacent_matrix(edge_initial)
        glonum = self.global_numberlists(0)
        specifics_a = ["B1","N01", "N10", "N11"]
        specifics_b = ["D11", "D12", "D13", "K1"]
        s_glonum_a, s_glonum_b = [], []
        # remove specifics
        for s in specifics_a:
            if s in glonum.keys():
                s_glonum_a.extend(glonum[s])
        for r in specifics_b:
            if r in glonum.keys():
                s_glonum_b.extend(glonum[r])
        for i, j in itertools.product(s_glonum_a, s_glonum_b):
            adjacent_matrix[i,j] = 0
            adjacent_matrix[j,i] = 0
        # get connected components
        csr = csr_matrix(adjacent_matrix)
        n_comp, labels = connected_components(csgraph=csr, directed=False, return_labels=True)
        assert n_comp == np.sum(self.molcount)
        protdict, prottype = {}, {}
        for lab in range(0, len(labels)):
            if labels[lab] not in protdict:
                protdict[labels[lab]] = []
            protdict[labels[lab]].append(lab) # add global number 
        for pro in protdict.keys():
            for m in range(0, len(self.molchar)):
                tag = glonum[self.molchar[m]]
                if len(set(protdict[pro])&set(tag)) != 0:
                    if self.mlists[m] not in prottype:
                        prottype[self.mlists[m]] = []
                    prottype[self.mlists[m]].append(pro)
        for m in range(0, len(self.mlists)):
            assert len(prottype[self.mlists[m]]) == self.molcount[m]
        return n_comp, labels, protdict, prottype, adjacent_matrix

    # retrieve connectivity of each molecules
    def protein_adjacent_matrix(self, tim):
        edge_current = self.connected_edge(tim)
        n_comp, labels, protdict, prottype, adj_native = self.global_protlists()
        glonum_current = self.global_numberlists(tim)
        adj_current = self.particle_adjacent_matrix(edge_current)
        adj_subtract = adj_current - adj_native
        protein_adj = np.zeros([n_comp, n_comp])
        bondindex = np.array(np.where(adj_subtract>0))
        if len(bondindex) != 0:
            for b in range(0, bondindex.shape[1]):
                protein_adj[labels[bondindex[0,b]], labels[bondindex[1,b]]] += 1
        assert np.array_equal(protein_adj, protein_adj.T)
        # align by molecular types
        #print(protdict, prottype)
        new_protein_adj = np.zeros((protein_adj.shape[0], protein_adj.shape[1]))
        binding = np.array(np.nonzero(protein_adj), dtype=int)
        index_temp = []
        for mol in self.mlists:
            index_temp.extend(prottype[mol])
        for b in range(0, binding.shape[1]):
            molnum1 = index_temp.index(binding[0,b])
            molnum2 = index_temp.index(binding[1,b])
            new_protein_adj[molnum1, molnum2] = 1
            new_protein_adj[molnum2, molnum1] = 1
        #print(index_temp)
        assert np.array_equal(new_protein_adj, new_protein_adj.T)
        return new_protein_adj

    def plot_prot_adj(self, tim):
        adj = self.protein_adjacent_matrix(tim)
        plt.imshow(adj, cmap="GnBu")
        plt.colorbar()
        plt.savefig("protein_adjmat.png")
        plt.show()
        return None

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

