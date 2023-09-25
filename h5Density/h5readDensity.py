import numpy as np
import matplotlib.pyplot as plt
import h5py
import hdf5plugin
import itertools
from tqdm import tqdm
import sys
import matplotlib.animation as animation

############### CAUTION:BEFORE USE ###############

# install hdf5plugin, readdy, h5py before use
# activate readdy environment by conda:
#     source activate readdy

##################################################

np.set_printoptions(threshold=np.inf)

def size(aa):
    radius = 0.1*2.24*(aa **0.392)
    return radius 

class h5readDensity: 
    def __init__(self, filename):
        f = h5py.File(filename, 'r')
        self.filename = filename
        self.traj = f['readdy']['trajectory']
        self.limrec = self.traj['limits']
        self.rec = self.traj['records']
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
        self.phaselim = 1.0 # width of phase bins
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
        # count nonzero value of n_molecule
        self.conc_result = np.zeros((len(self.limrec), 4))
        self.variables = []
   
    # time iteration(loop this function)
    def calc_dilute(self, tim, separate=True):
        # settings
        criteria = 15 # when particles in the certain z point, we treat the phase it as "condensed phase"
        record = self.rec[self.limrec[tim, 0]:self.limrec[tim, 1]]
        distrec = np.zeros((len(record), 2))
        amprec = np.zeros(self.n_ampar)
        psdrec = np.zeros(self.n_psd95)
        nmdrec = np.zeros(self.n_nmdar)
        camrec = np.zeros(self.n_camk2)
        x = np.arange(-self.zbox/2, self.zbox/2, self.phaselim)
        ruiseki = np.zeros(x.shape)
        amp, psd, nmd, cam = 0, 0, 0, 0
        for r in range(0, len(record)):
            # in each particle, see particle type by record[r][0] and find it in
            radii = self.size_list[np.where(self.types_num==record[r][0])[0][0]]
            distrec[r,:] = np.array([record[r][3][2]-radii, record[r][3][2]+radii])
            if self.n_ampar != 0 and record[r][0] == self.types_num[self.types_str.index('A')]:
                amprec[amp] = record[r][3][2]
                amp += 1
            if self.n_psd95 != 0 and record[r][0] == self.types_num[self.types_str.index('E')]:
                psdrec[psd] = record[r][3][2]
                psd += 1
            if self.n_nmdar != 0 and record[r][0] == self.types_num[self.types_str.index('G')]:
                nmdrec[nmd] = record[r][3][2]
                nmd += 1
            if self.n_camk2 != 0 and record[r][0] == self.types_num[self.types_str.index('M')]:
                camrec[cam] = record[r][3][2]
                cam += 1
            for dx in range(0, len(x)):
                if x[dx] > distrec[r,0] and x[dx] < distrec[r,1]:
                    ruiseki[dx] +=1
        ## if True, separate dilte and condensed phase
        if separate == True:
            try:
                # phase0 -> 0, phase1 -> 1
                phase = np.where(np.copy(ruiseki)>0, 1, 0)
                phasegroup = [(k, list(g)) for k, g in itertools.groupby(phase)]
                phaseval = [phasegroup[i][0] for i in range(0, len(phasegroup))]
                phaselists = np.cumsum([len(phasegroup[k][1]) for k in range(0, len(phasegroup))])
                delete_lists = []
                # in case dilute phase is too thin, delete the dilute phase
                for p in range(1, len(phaselists)):
                    if phaseval[p] == 0 and (phaselists[p]-phaselists[p-1]) < 5*self.phaselim:
                        delete_lists.extend([p-1, p])
                phaselists = np.delete(phaselists, delete_lists)
                condensed_phase = []
                dilute_phase = []
                dilute_z = self.zbox
                if phase[0] == 1 and phase[-1] == 1:
                    if np.max(ruiseki[0:phaselists[0]]) > criteria or np.max(ruiseki[phaselists[-2]:phaselists[-1]]) > criteria:
                        condensed_phase.append([-self.zbox/2, phaselists[0]*self.phaselim-self.zbox/2])
                        condensed_phase.append([phaselists[-2]*self.phaselim-self.zbox/2, self.zbox/2])
                        dilute_z -= (phaselists[0]+phaselists[-1]-phaselists[-2])*self.phaselim
                    else:
                        dilute_phase.append([-self.zbox/2, phaselists[0]*self.phaselim-self.zbox/2])
                        dilute_phase.append([phaselists[-2]*self.phaselim-self.zbox/2, self.zbox/2])
                for p in range(1, len(phaselists)):
                    if np.max(ruiseki[phaselists[p-1]:phaselists[p]]) > criteria:
                        if [phaselists[p-1]*self.phaselim-self.zbox/2, phaselists[p]*self.phaselim-self.zbox/2] in condensed_phase:
                            pass
                        else:
                            condensed_phase.append([phaselists[p-1]*self.phaselim-self.zbox/2, phaselists[p]*self.phaselim-self.zbox/2])
                            dilute_z -= (phaselists[p]-phaselists[p-1])*self.phaselim
                    else:
                        dilute_phase.append([phaselists[p-1]*self.phaselim-self.zbox/2, phaselists[p]*self.phaselim-self.zbox/2])
                # if dilute phase is too thin, phase near to the thin dilute phase belongs to condensed phase  
                # calculate the number of particle in dilute phase
                count_ampar = 0
                count_psd95 = 0
                count_nmdar = 0
                count_camk2 = 0
                for d in range(0, len(dilute_phase)):
                    count_ampar += np.count_nonzero((amprec > dilute_phase[d][0]) & (amprec < dilute_phase[d][1]))
                    count_psd95 += np.count_nonzero((psdrec > dilute_phase[d][0]) & (psdrec < dilute_phase[d][1]))
                    count_nmdar += np.count_nonzero((nmdrec > dilute_phase[d][0]) & (nmdrec < dilute_phase[d][1]))
                    count_camk2 += np.count_nonzero((camrec > dilute_phase[d][0]) & (camrec < dilute_phase[d][1]))
                scaling_factor = 10/(6.02214076*dilute_z*self.xbox*self.xbox)
                ampar_dilute_conc = count_ampar*scaling_factor
                psd95_dilute_conc = count_psd95*scaling_factor
                nmdar_dilute_conc = count_nmdar*scaling_factor
                camk2_dilute_conc = count_camk2*scaling_factor
                self.conc_result[tim,:] = [ampar_dilute_conc, psd95_dilute_conc, nmdar_dilute_conc, camk2_dilute_conc]
                self.variables = ruiseki, phase, condensed_phase, amprec, psdrec, nmdrec, camrec
            except Exception as e:
                self.conc_result[tim,:] = [float('nan'), float('nan'), float('nan'), float('nan')]
                print("ERROR: Could not separate phase into dilute and condensed") 
        else:
            self.variables = ruiseki, amprec, psdrec, nmdrec, camrec
        return None

    # record the calculated dilute phase concentration
    def track_dilute(self):
        for tim in tqdm(range(0, len(self.limrec))):
            self.calc_dilute(tim, separate=True)
        np.savetxt("dilutetraj_"+self.filename+".txt", self.conc_result)
        conc_result_mean = np.mean(self.conc_result[:], axis=0)
        with open("output"+self.filename+".txt", "w") as a:
            a.writelines([self.filename, " ", str(conc_result_mean[0]), " ", str(conc_result_mean[1]), "\n"])
        print("====================")
        print("CALCULATION FINISHED")
        print("====================")
        return None
    
    # drawing picture 
    def plot_dilute(self, tim, sep, picname):
        self.calc_dilute(tim, separate=sep)
        if sep == True:
            ruiseki, phaselists, condensed_phase, amprec, psdrec, nmdrec, camrec = self.variables
        else:
            ruiseki, amprec, psdrec, nmdrec, camrec = self.variables
        fig = plt.figure(figsize=[20.,5.0])
        ax = fig.add_subplot(111)
        # draw particle number histogram
        ampar_hist, ampar_bins = np.histogram(amprec, bins=100, range=(-self.zbox/2, self.zbox/2))
        psd95_hist, psd95_bins = np.histogram(psdrec, bins=100, range=(-self.zbox/2, self.zbox/2))
        nmdar_hist, nmdar_bins = np.histogram(nmdrec, bins=100, range=(-self.zbox/2, self.zbox/2))
        camk2_hist, camk2_bins = np.histogram(camrec, bins=100, range=(-self.zbox/2, self.zbox/2))
        if sep == True:
            for c in range(0, len(condensed_phase)):
                ax.axvspan(condensed_phase[c][0], condensed_phase[c][1], color = "orange", alpha=0.1)
            ax1 = ax.twinx()
            x = np.arange(-self.zbox/2, self.zbox/2, self.phaselim)
            ax1.fill_between(x, ruiseki, np.zeros(len(x)), alpha=0.3)
            ax1.set_xlim(-self.zbox/2, self.zbox/2)
            ax1.set_ylim(0, 110)
        ax.set_xlim(-self.zbox/2, self.zbox/2)
        ax.plot(ampar_bins[:-1]+5*0.5, ampar_hist, color="red")
        ax.plot(psd95_bins[:-1]+5*0.5, psd95_hist, color="blue")
        ax.plot(nmdar_bins[:-1]+5*0.5, nmdar_hist, color="magenta")
        ax.plot(camk2_bins[:-1]+5*0.5, camk2_hist, color="green")
        ax.set_ylim(0, 15)
        plt.savefig(str(picname))
        plt.show()
        return None

    def animate_dilute(self, start, stop, skip, animname):
        fig = plt.figure(figsize=[20.,5.0])
        ax01 = fig.add_subplot(111)
        frames = []
        x = np.arange(-self.zbox/2, self.zbox/2, self.phaselim)
        ax01.set_xlim(-self.zbox/2, self.zbox/2)
        ax01.set_ylim(0, 15)
        for i in tqdm(range(start, stop, skip)):
            self.calc_dilute(i, separate=False)
            ruiseki, amprec, psdrec, nmdrec, camrec = self.variables
            # draw particle number histogram
            ampar_hist, ampar_bins = np.histogram(amprec, bins=100, range=(-self.zbox/2, self.zbox/2))
            psd95_hist, psd95_bins = np.histogram(psdrec, bins=100, range=(-self.zbox/2, self.zbox/2))
            nmdar_hist, nmdar_bins = np.histogram(nmdrec, bins=100, range=(-self.zbox/2, self.zbox/2))
            camk2_hist, camk2_bins = np.histogram(camrec, bins=100, range=(-self.zbox/2, self.zbox/2))
            img1 = ax01.plot(ampar_bins[:-1]+5*0.5, ampar_hist, color="red")
            img2 = ax01.plot(psd95_bins[:-1]+5*0.5, psd95_hist, color="blue")
            img3 = ax01.plot(nmdar_bins[:-1]+5*0.5, nmdar_hist, color="magenta")
            img4 = ax01.plot(camk2_bins[:-1]+5*0.5, camk2_hist, color="green")
            txt = ax01.text(150, 13, "time = "+str(i*8000*0.25/1e6)+" ms", size=20)
            frame = img1 + img2 + img3 + img4 + [txt]
            frames.append(frame)
        ani = animation.ArtistAnimation(fig, frames, interval=100)
        ani.save(animname, writer="pillow")
        plt.show()
        return print("Movie Generation Completed")

#result1 = h5readDensity("PSD4_slab_activated.h5")
#result1.animate_dilute(0, 10000, 100, "activated.gif")
#result2 = h5readDensity("PSD4_slab_inactivated.h5")
#result2.animate_dilute(0, 10000, 100, "inactivated.gif")
#result1.plot_dilute(10000, False, "activated10000.png")
#result2.plot_dilute(10000, False, "inactivated10000.png")

