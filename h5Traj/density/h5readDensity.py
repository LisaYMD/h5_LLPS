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

from .. import h5readTraj

def size(aa):
    radius = 0.1*2.24*(aa **0.392)
    return radius 

class h5readDensity( h5readTraj ): 
    def __init__(self, filename, dim=3):
        super().__init__(filename, 3)
        self.phaselim = 1.0 # width of phase bins
        # count nonzero value of n_molecule
        self.conc_result = np.zeros((len(self.limrec), len(self.mlists)))
        self.variables = []
        return None
   
    # time iteration(loop this function)
    def calc_dilute(self, tim, separate=True):
        # settings
        moltypecount = len(self.mlists)
        criteria = 15 # when particles in the certain z point, we treat the phase it as "condensed phase"
        record = self.rec[self.limrec[tim, 0]:self.limrec[tim, 1]]
        distrec = np.zeros((len(record), 2))
        molrec = np.ones([moltypecount, int(np.max(self.molcount))])*(-10*self.zbox)
        x = np.arange(-self.zbox/2, self.zbox/2, self.phaselim)
        ruiseki = np.zeros(x.shape)
        ctl = np.zeros(moltypecount)
        for r in range(0, len(record)):
            # in each particle, see particle type by record[r][0] and find it in
            radii = self.size_list[np.where(self.types_num==record[r][0])[0][0]]
            distrec[r,:] = np.array([record[r][3][2]-radii, record[r][3][2]+radii])
            for mol in range(0, moltypecount):
                if self.molcount[mol] != 0 and record[r][0] == self.types_num[self.types_str.index(self.molchar[mol])]:
                    molrec[mol, int(ctl[mol])] = record[r][3][2]
                    ctl[mol] += 1
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
                else:
                    dilute_phase.append([-self.zbox/2, phaselists[0]*self.phaselim-self.zbox/2])
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
                count_mols = np.zeros(moltypecount)
                for d in range(0, len(dilute_phase)):
                    for p in range(0, moltypecount):
                        molrec2 = molrec[p][np.abs(molrec[p])<self.zbox*9] 
                        assert len(molrec2) == self.molcount[p]
                        count_mols[p] += np.count_nonzero((molrec2 > dilute_phase[d][0]) & (molrec2 < dilute_phase[d][1]))
                scaling_factor = 10/(6.02214076*dilute_z*self.xbox*self.xbox)
                mol_dilute_conc = np.zeros(moltypecount)
                for p in range(0, moltypecount):
                    mol_dilute_conc[p] = count_mols[p] * scaling_factor
                self.conc_result[tim,:] = mol_dilute_conc
                self.variables = ruiseki, phase, condensed_phase, molrec
            except Exception as e:
                self.conc_result[tim,:] = [float('nan'), float('nan'), float('nan'), float('nan')]
                print("ERROR: Could not separate phase into dilute and condensed") 
        else:
            self.variables = ruiseki, molrec
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
            ruiseki, phaselists, condensed_phase, molrec = self.variables
        else:
            ruiseki, molrec = self.variables
        fig = plt.figure(figsize=[20.,5.0])
        ax = fig.add_subplot(111)
        # draw particle number histogram
        for p in range(0, len(self.mlists)):
            molrec3 = molrec[p][np.abs(molrec[p])<self.zbox/9]
            assert len(molrec3) == self.molcount[p]
            hist, bins = np.histogram(molrec3, bins=200, range=(-self.zbox/2, self.zbox/2))
            ax.plot(bins[:-1]+5*0.5, hist, color="black")
        if sep == True:
            for c in range(0, len(condensed_phase)):
                ax.axvspan(condensed_phase[c][0], condensed_phase[c][1], color = "orange", alpha=0.1)
            ax1 = ax.twinx()
            x = np.arange(-self.zbox/2, self.zbox/2, self.phaselim)
            ax1.fill_between(x, ruiseki, np.zeros(len(x)), alpha=0.3)
            ax1.set_xlim(-self.zbox/2, self.zbox/2)
            ax1.set_ylim(0, 110)
        ax.set_xlim(-self.zbox/2, self.zbox/2)
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
            ruiseki, molrec = self.variables
            # draw particle number histogram
            ampar_hist, ampar_bins = np.histogram(molrec[0], bins=200, range=(-self.zbox/2, self.zbox/2))
            psd95_hist, psd95_bins = np.histogram(molrec[1], bins=200, range=(-self.zbox/2, self.zbox/2))
            nmdar_hist, nmdar_bins = np.histogram(molrec[2], bins=200, range=(-self.zbox/2, self.zbox/2))
            camk2_hist, camk2_bins = np.histogram(molrec[3], bins=200, range=(-self.zbox/2, self.zbox/2))
            img1 = ax01.plot(ampar_bins[:-1]+5*0.5, ampar_hist, color="red")
            img2 = ax01.plot(psd95_bins[:-1]+5*0.5, psd95_hist, color="blue")
            img3 = ax01.plot(nmdar_bins[:-1]+5*0.5, nmdar_hist, color="magenta")
            img4 = ax01.plot(camk2_bins[:-1]+5*0.5, camk2_hist, color="green")
            txt = ax01.text(150, 13, "time = "+str(i*2500*0.25/1e6)+" ms", size=20)
            frame = img1 + img2 + img3 + img4 + [txt]
            frames.append(frame)
        ani = animation.ArtistAnimation(fig, frames, interval=100)
        ani.save(animname, writer="pillow")
        plt.show()
        return print("Movie Generation Completed")

