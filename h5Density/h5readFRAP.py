import numpy as np
import matplotlib.pyplot as plt
import h5py
import hdf5plugin
import itertools
from tqdm import tqdm
from scipy import stats
import matplotlib
import sys
############### CAUTION:BEFORE USE ###############

# install hdf5plugin, readdy, h5py before use
# activate readdy environment by conda:
#     source activate readdy

##################################################

from h5readDensity import h5readDensity

class h5readFRAP( h5readDensity ):
    # self: filename, traj, self.limrec, rec, types_str, types_num, xbox, zbox, sizelist, phaselim
    #       n_ampar, n_psd95, n_nmdar, n_camk2, mlists, conc_result, variables
    def __init__(self, filename):
        super().__init__(filename)

    def frap_iteration(self, tim):
        #self.conc_result = np.zeros((len(self.limrec), 2))
        center_of_newamp = np.zeros(len(self.limrec))
        center_of_total = np.zeros(len(self.limrec))
        super().calc_dilute(tim, separate=True)
        ### same as calc_dilute()
        record = self.rec[self.limrec[tim, 0]:self.limrec[tim, 1]]
        oldamp = []
        newamp = []
        oldamp_number = []
        newamp_number = []
        amp, psd = 0, 0
        _, _, condensed_phase, amprec, psdrec, nmdrec, camrec = self.variables
        ### # different from ordinary dilute calculation
        ### pick up the leftmost 1/4 z position of condensed phase
        condensed_lim = (3*condensed_phase[0][0]+1*condensed_phase[0][1])/4
        for r in range(0, len(record)):
            if record[r][0] == self.types_num[self.types_str.index('A')]:
                if record[r][3][2] < (3*condensed_phase[0][0]+1*condensed_phase[0][1])/4:
                    amprec[amp] = record[r][3][2]
                    amp += 1
                    newamp.append(record[r][3][2])
                    newamp_number.append(record[r][1])
                else:
                    amprec[amp] = record[r][3][2]
                    amp += 1
                    oldamp.append(record[r][3][2])
                    oldamp_number.append(record[r][1])
            if record[r][0] == self.types_num[self.types_str.index('F')]:
                psdrec[psd] = record[r][3][2]
                psd += 1
        return oldamp, newamp, oldamp_number, newamp_number

    def frap_plot(self, tim):
        plt.rcParams["font.family"] = "Arial"
        plt.rcParams["font.size"] = 9
        plt.rcParams["xtick.direction"] = "in"
        plt.rcParams["ytick.direction"] = "in"
        plt.rcParams["xtick.major.width"] = 1.0 
        plt.rcParams["ytick.major.width"] = 1.0 
        plt.rcParams["xtick.minor.width"] = 1.0 
        plt.rcParams["ytick.minor.width"] = 1.0 
        plt.rcParams["xtick.major.size"] = 5
        plt.rcParams["ytick.major.size"] = 5
        plt.rcParams["xtick.minor.size"] = 5
        plt.rcParams["ytick.minor.size"] = 5
        plt.rcParams['xtick.top'] = False
        plt.rcParams['ytick.right'] = False
        plt.rcParams["mathtext.fontset"] = 'custom'
        plt.rcParams['mathtext.default'] = 'rm'
        palette = ["#d43f3a", "#eea236", "#5cb85c", "#46b8da", "#357ebd", "#9632b8", "#b8b8b8"]
        amprec, psdrec, oldamp, newamp = self.frap_iteration(tim)
        _, _, condensed_phase, _, _, _, _ = self.variables
        fig = plt.figure(figsize=[5., 1])
        ax = fig.add_subplot(111)
        ax.spines.top.set_visible(False)
        xx = np.linspace(-self.zbox/2, self.zbox/2, 1000)
        # draw particle number histogram
        ampar_hist, ampar_bins = np.histogram(amprec, bins=100, range=(-self.zbox/2, self.zbox/2))
        psd95_hist, psd95_bins = np.histogram(psdrec, bins=100, range=(-self.zbox/2, self.zbox/2))
        oldamp_hist, oldamp_bins = np.histogram(oldamp, bins=100, range=(-self.zbox/2, self.zbox/2))
        newamp_hist, newamp_bins = np.histogram(newamp, bins=100, range=(-self.zbox/2, self.zbox/2))
        half = (condensed_phase[0][0]+condensed_phase[0][1])/2
        ax1 = ax.twinx()
        x = np.arange(-self.zbox/2, self.zbox/2, self.phaselim)
        ax.set_xlim(-self.zbox/2, self.zbox/2)
        ax1.set_xlim(-self.zbox/2, self.zbox/2)
        kde_newamp = stats.gaussian_kde(newamp)
        kde_total = stats.gaussian_kde(amprec)
        ax1.plot(xx, kde_total(xx), color=palette[0], label="total AMPAR")
        ax.fill_between(ampar_bins[:-1], ampar_hist, step="post", color=palette[0], alpha=0.5)
        ax1.plot(xx, kde_newamp(xx), color=palette[4], label="bleached AMPAR")
        ax.fill_between(newamp_bins[:-1], newamp_hist, step="post", color=palette[4], alpha=0.5)
        ax.set_xlabel("Z position", fontsize=11)
        ax.set_ylabel("Number", fontsize=11)
        ax1.set_ylabel("frequency", fontsize=11)
        #print(ampar_hist, ampar_bins, oldamp_hist, oldamp_bins)
        plt.savefig("slab_frap"+str(tim)+".svg")
        plt.show()
        return None

    def frap_result(self, duration, smooth_time):
        newamp_center = np.zeros((duration, smooth_time))
        oldamp_center = np.zeros((duration, smooth_time))
        for s in tqdm(range(0, smooth_time)): # starting time
            oldamp, newamp, oldamp_number, newamp_number = self.frap_iteration(s)
            for d in range(0, duration):
                rec_newamp = []
                rec_oldamp = []
                record = self.rec[self.limrec[s+d, 0]:self.limrec[s+d, 1]]
                for r in range(0, len(record)):
                    if record[r][0] == self.types_num[self.types_str.index('A')]:
                        if record[r][1] in newamp_number:
                            rec_newamp.append(record[r][3][2])
                        elif record[r][1] in oldamp_number:
                            rec_oldamp.append(record[r][3][2])
                        else:
                            print("Error")
                newamp_center[d, s] = np.mean(rec_newamp)
                oldamp_center[d, s] = np.mean(rec_oldamp)
        np.savetxt("newamp_center.txt", newamp_center)
        np.savetxt("oldamp_center.txt", oldamp_center)
        sub = np.mean(oldamp_center-newamp_center, axis=1)
        return sub

    def frap_trajectory(self):
        plt.rcParams["font.family"] = "Arial"
        plt.rcParams["font.size"] = 9
        plt.rcParams["xtick.direction"] = "in"
        plt.rcParams["ytick.direction"] = "in"
        plt.rcParams["xtick.major.width"] = 1.0 
        plt.rcParams["ytick.major.width"] = 1.0 
        plt.rcParams["xtick.minor.width"] = 1.0 
        plt.rcParams["ytick.minor.width"] = 1.0 
        plt.rcParams["xtick.major.size"] = 5
        plt.rcParams["ytick.major.size"] = 5
        plt.rcParams["xtick.minor.size"] = 5
        plt.rcParams["ytick.minor.size"] = 5
        plt.rcParams['xtick.top'] = False
        plt.rcParams['ytick.right'] = False
        plt.rcParams["mathtext.fontset"] = 'custom'
        plt.rcParams['mathtext.default'] = 'rm'
        palette = ["#d43f3a", "#eea236", "#5cb85c", "#46b8da", "#357ebd", "#9632b8", "#b8b8b8"]
        subtraction = self.frap_result(4000,1000)
        fig = plt.figure(figsize=[5,2])
        ax = fig.add_subplot(111)
        ax.set_xlabel(r"$\Delta$ Time (μs)", fontsize=11)
        ax.set_ylabel(r"$\Delta Z $ (nm)", fontsize=11)
        ax.plot(np.arange(0, len(subtraction))*8000*0.25/1e3, subtraction, color=palette[5])
        ax.axhline(y=0, linestyle="dashed", color=palette[6])
        plt.savefig("frap_subtraction.svg")
        plt.show()
        return None

#result = h5readFRAP("strictampar_cutoff1.6_2.h5")


