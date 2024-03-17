import numpy as np
import matplotlib.pyplot as plt
import h5py
import hdf5plugin
import itertools
from tqdm import tqdm
import sys
############### CAUTION:BEFORE USE ###############

# install hdf5plugin, readdy, h5py before use
# activate readdy environment by conda:
#     source activate readdy

##################################################

from .h5readDensity import h5readDensity

class h5readMSD( h5readDensity ):
    # self: filename, traj, limrec, rec, types_str, types_num, xbox, zbox, sizelist, phaselim
    #       n_ampar, n_psd95, n_nmdar, n_camk2, mlists, conc_result, variables
    def __init__(self, filename):
        super().__init__(filename)

    def calc_dilute_withxyz(self, tim): 
        # settings
        criteria = 15 # when particles in the certain z point, we treat the phase it as "condensed phase"
        record = self.rec[self.limrec[tim, 0]:self.limrec[tim, 1]]
        distrec = np.zeros((len(record), 2))
        amprec = np.zeros((self.n_ampar, 5))  # [global no, xpos, ypos, zpos, dilute(0)/condensed(1)]
        psdrec = np.zeros((self.n_psd95, 5))
        nmdrec = np.zeros((self.n_nmdar, 5))
        camrec = np.zeros((self.n_camk2, 5))
        x = np.arange(-self.zbox/2, self.zbox/2, self.phaselim)
        ruiseki = np.zeros(x.shape) 
        amp, psd, nmd, cam = 0, 0, 0, 0
        for r in range(0, len(record)):
            # in each particle, see particle type by record[r][0] and find it in
            radii = self.size_list[np.where(self.types_num==record[r][0])[0][0]]
            distrec[r,:] = np.array([record[r][3][2]-radii, record[r][3][2]+radii])
            if self.n_ampar != 0 and record[r][0] == self.types_num[self.types_str.index('A')]:
                amprec[amp,:] = np.array([record[r][1], record[r][3][0], record[r][3][1], record[r][3][2], 0])
                amp += 1
            if self.n_psd95 != 0 and record[r][0] == self.types_num[self.types_str.index('E')]:
                psdrec[psd,:] = np.array([record[r][1], record[r][3][0], record[r][3][1], record[r][3][2], 0])
                psd += 1
            if self.n_nmdar != 0 and record[r][0] == self.types_num[self.types_str.index('G')]:
                nmdrec[nmd,:] = np.array([record[r][1], record[r][3][0], record[r][3][1], record[r][3][2], 0])
                nmd += 1
            if self.n_camk2 != 0 and record[r][0] == self.types_num[self.types_str.index('M')]:
                camrec[cam,:] = np.array([record[r][1], record[r][3][0], record[r][3][1], record[r][3][2], 0])
                cam += 1
            for dx in range(0, len(x)):
                if x[dx] > distrec[r,0] and x[dx] < distrec[r,1]:
                    ruiseki[dx] +=1
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
                count_ampar += np.count_nonzero((amprec[:,3] > dilute_phase[d][0]) & (amprec[:,3] < dilute_phase[d][1]))
                count_psd95 += np.count_nonzero((psdrec[:,3] > dilute_phase[d][0]) & (psdrec[:,3] < dilute_phase[d][1]))
                count_nmdar += np.count_nonzero((nmdrec[:,3] > dilute_phase[d][0]) & (nmdrec[:,3] < dilute_phase[d][1]))
                count_camk2 += np.count_nonzero((camrec[:,3] > dilute_phase[d][0]) & (camrec[:,3] < dilute_phase[d][1]))
            scaling_factor = 10/(6.02214076*dilute_z*self.xbox*self.xbox)
            ampar_dilute_conc = count_ampar*scaling_factor
            psd95_dilute_conc = count_psd95*scaling_factor
            nmdar_dilute_conc = count_nmdar*scaling_factor
            camk2_dilute_conc = count_camk2*scaling_factor
            self.conc_result[tim,:] = [ampar_dilute_conc, psd95_dilute_conc, nmdar_dilute_conc, camk2_dilute_conc]
            for c in range(0, len(condensed_phase)):
                for p in range(0, self.n_ampar):
                    if amprec[p,3] > condensed_phase[c][0] and amprec[p,3] < condensed_phase[c][1]:
                        amprec[p,4] = 1
                for p in range(0, self.n_psd95):
                    if psdrec[p,3] > condensed_phase[c][0] and psdrec[p,3] < condensed_phase[c][1]:
                        psdrec[p,4] = 1
                for p in range(0, self.n_nmdar):
                    if nmdrec[p,3] > condensed_phase[c][0] and nmdrec[p,3] < condensed_phase[c][1]:
                        nmdrec[p,4] = 1
                for p in range(0, self.n_camk2):
                    if camrec[p,3] > condensed_phase[c][0] and camrec[p,3] < condensed_phase[c][1]:
                        camrec[p,4] = 1
            self.variables = ruiseki, phase, condensed_phase, amprec, psdrec, nmdrec, camrec 
        except Exception as e:
                self.conc_result[tim,:] = [float('nan'), float('nan'), float('nan'), float('nan')]
                print("ERROR: Could not separate phase into dilute and condensed") 
        return None

    # before calculating MSD, convert trajectory as linear growth (required for PBC)
    def convert_trajectory(self, duration):
        amprec = np.zeros((duration, self.n_ampar, 4))  # [global no, xpos, ypos, zpos, dilute(0)/condensed(1)]
        psdrec = np.zeros((duration, self.n_psd95, 4))
        nmdrec = np.zeros((duration, self.n_nmdar, 4))  # [global no, xpos, ypos, zpos, dilute(0)/condensed(1)]
        camrec = np.zeros((duration, self.n_camk2, 4))
        for tim in tqdm(range(0, duration)):
            record = self.rec[self.limrec[tim, 0]:self.limrec[tim, 1]]
            amp, psd, nmd, cam = 0, 0, 0, 0
            for r in range(0, len(record)):
                if tim == 0:
                    if self.n_ampar != 0 and record[r][0] == self.types_num[self.types_str.index('A')]:
                        amprec[tim, amp,:] = np.array([record[r][1], record[r][3][0], record[r][3][1], record[r][3][2]])
                        amp += 1
                    if self.n_psd95 != 0 and record[r][0] == self.types_num[self.types_str.index('F')]:
                        psdrec[tim, psd,:] = np.array([record[r][1], record[r][3][0], record[r][3][1], record[r][3][2]])
                        psd += 1
                    if self.n_nmdar != 0 and record[r][0] == self.types_num[self.types_str.index('G')]:
                        nmdrec[tim, nmd,:] = np.array([record[r][1], record[r][3][0], record[r][3][1], record[r][3][2]])
                        nmd += 1
                    if self.n_camk2 != 0 and record[r][0] == self.types_num[self.types_str.index('M')]:
                        camrec[tim, cam,:] = np.array([record[r][1], record[r][3][0], record[r][3][1], record[r][3][2]])
                        cam += 1
                else:
                    if self.n_ampar != 0 and record[r][0] == self.types_num[self.types_str.index('A')]:
                        axpos = record[r][3][0]
                        aypos = record[r][3][1]
                        azpos = record[r][3][2]
                        axsub = record[r][3][0] - amprec[tim-1, amp, 1]
                        aysub = record[r][3][1] - amprec[tim-1, amp, 2]
                        azsub = record[r][3][2] - amprec[tim-1, amp, 3]
                        if axsub > self.xbox/2:
                            axpos -= self.xbox*(np.abs(axsub-self.xbox/2)//self.xbox + 1)
                        elif axsub < -self.xbox/2:
                            axpos += self.xbox*(np.abs(axsub+self.xbox/2)//self.xbox + 1)
                        else:
                            pass
                        if aysub > self.xbox/2:
                            aypos -= self.xbox*(np.abs(aysub-self.xbox/2)//self.xbox + 1)
                        elif aysub < -self.xbox/2:
                            aypos += self.xbox*(np.abs(aysub+self.xbox/2)//self.xbox + 1)
                        else:
                            pass
                        if azsub > self.zbox/2:
                            azpos -= self.zbox*(np.abs(azsub-self.zbox/2)//self.zbox + 1)
                        elif azsub < -self.zbox/2:
                            azpos += self.zbox*(np.abs(azsub+self.zbox/2)//self.zbox + 1)
                        else:
                            pass
                        amprec[tim, amp,:] = np.array([record[r][1], axpos, aypos, azpos])
                        amp += 1
                    if self.n_psd95 != 0 and record[r][0] == self.types_num[self.types_str.index('E')]:
                        pxpos = record[r][3][0]
                        pypos = record[r][3][1]
                        pzpos = record[r][3][2]
                        pxsub = record[r][3][0] - psdrec[tim-1, psd, 1]
                        pysub = record[r][3][1] - psdrec[tim-1, psd, 2]
                        pzsub = record[r][3][2] - psdrec[tim-1, psd, 3]
                        if pxsub > self.xbox/2:
                            pxpos -= self.xbox*(np.abs(pxsub-self.xbox/2)//self.xbox + 1)
                        elif pxsub < -self.xbox/2:
                            pxpos += self.xbox*(np.abs(pxsub+self.xbox/2)//self.xbox + 1)
                        else:
                            pass
                        if pysub > self.xbox/2:
                            pypos -= self.xbox*(np.abs(pysub-self.xbox/2)//self.xbox + 1)
                        elif pysub < -self.xbox/2:
                            pypos += self.xbox*(np.abs(pysub+self.xbox/2)//self.xbox + 1)
                        else:
                            pass
                        if pzsub > self.zbox/2:
                            pzpos -= self.zbox*(np.abs(pzsub-self.zbox/2)//self.zbox + 1)
                        elif pzsub < -self.zbox/2:
                            pzpos += self.zbox*(np.abs(pzsub+self.zbox/2)//self.zbox + 1)
                        else:
                            pass
                        psdrec[tim, psd,:] = np.array([record[r][1], pxpos, pypos, pzpos])
                        psd += 1
                    if self.n_nmdar != 0 and record[r][0] == self.types_num[self.types_str.index('G')]:
                        nxpos = record[r][3][0]
                        nypos = record[r][3][1]
                        nzpos = record[r][3][2]
                        nxsub = record[r][3][0] - nmdrec[tim-1, nmd, 1]
                        nysub = record[r][3][1] - nmdrec[tim-1, nmd, 2]
                        nzsub = record[r][3][2] - nmdrec[tim-1, nmd, 3]
                        if nxsub > self.xbox/2:
                            nxpos -= self.xbox*(np.abs(nxsub-self.xbox/2)//self.xbox + 1)
                        elif nxsub < -self.xbox/2:
                            nxpos += self.xbox*(np.abs(nxsub+self.xbox/2)//self.xbox + 1)
                        else:
                            pass
                        if nysub > self.xbox/2:
                            nypos -= self.xbox*(np.abs(nysub-self.xbox/2)//self.xbox + 1)
                        elif nysub < -self.xbox/2:
                            nypos += self.xbox*(np.abs(nysub+self.xbox/2)//self.xbox + 1)
                        else:
                            pass
                        if nzsub > self.zbox/2:
                            nzpos -= self.zbox*(np.abs(nzsub-self.zbox/2)//self.zbox + 1)
                        elif nzsub < -self.zbox/2:
                            nzpos += self.zbox*(np.abs(nzsub+self.zbox/2)//self.zbox + 1)
                        else:
                            pass
                        nmdrec[tim, nmd,:] = np.array([record[r][1], nxpos, nypos, nzpos])
                        nmd += 1
                    if self.n_camk2 != 0 and record[r][0] == self.types_num[self.types_str.index('M')]:
                        cxpos = record[r][3][0]
                        cypos = record[r][3][1]
                        czpos = record[r][3][2]
                        cxsub = record[r][3][0] - camrec[tim-1, cam, 1]
                        cysub = record[r][3][1] - camrec[tim-1, cam, 2]
                        czsub = record[r][3][2] - camrec[tim-1, cam, 3]
                        if cxsub > self.xbox/2:
                            cxpos -= self.xbox*(np.abs(cxsub-self.xbox/2)//self.xbox + 1)
                        elif cxsub < -self.xbox/2:
                            cxpos += self.xbox*(np.abs(cxsub+self.xbox/2)//self.xbox + 1)
                        else:
                            pass
                        if cysub > self.xbox/2:
                            cypos -= self.xbox*(np.abs(cysub-self.xbox/2)//self.xbox + 1)
                        elif cysub < -self.xbox/2:
                            cypos += self.xbox*(np.abs(cysub+self.xbox/2)//self.xbox + 1)
                        else:
                            pass
                        if czsub > self.zbox/2:
                            czpos -= self.zbox*(np.abs(czsub-self.zbox/2)//self.zbox + 1)
                        elif czsub < -self.zbox/2:
                            czpos += self.zbox*(np.abs(czsub+self.zbox/2)//self.zbox + 1)
                        else:
                            pass
                        camrec[tim, cam,:] = np.array([record[r][1], cxpos, cypos, czpos])
                        cam += 1
        return amprec, psdrec, nmdrec, camrec

    def track_trajectory(self, duration, particle_num):
        amprec_extend = np.zeros([duration, self.n_ampar, 4])
        psdrec_extend = np.zeros([duration, self.n_psd95, 4])
        nmdrec_extend = np.zeros([duration, self.n_nmdar, 4])
        camrec_extend = np.zeros([duration, self.n_camk2, 4]) 
        amprec_extend, psdrec_extend, nmdrec_extend, camrec_extend = self.convert_trajectory(duration)
        fig = plt.figure(figsize=[10.,10.])
        ax = fig.add_subplot(111)
        ax.axvline(x=-25, color="black")
        ax.axvline(x=25, color="black")
        ax.axhline(y=-25, color="black")
        ax.axhline(y=25, color="black")
        ax.axvline(x=-75, color="black")
        ax.axvline(x=75, color="black")
        ax.axhline(y=-75, color="black")
        ax.axhline(y=75, color="black")
        ax.axvline(x=-125, color="black")
        ax.axvline(x=125, color="black")
        ax.axhline(y=-125, color="black")
        ax.axhline(y=125, color="black")
        for i in range(0, particle_num):
            if particle_num < self.n_ampar:
                ax.plot(amprec_extend[:,i,1], amprec_extend[:,i,2], color="red")
            if particle_num < self.n_psd95: 
                ax.plot(psdrec_extend[:,i,1], psdrec_extend[:,i,2], color="blue")
            if particle_num < self.n_nmdar: 
                ax.plot(nmdrec_extend[:,i,1], nmdrec_extend[:,i,2], color="magenta")
            if particle_num < self.n_camk2:
                ax.plot(camrec_extend[:,i,1], camrec_extend[:,i,2], color="green")
        plt.savefig("trajectory_example.png")
        plt.show()
        return None 
    
    def calculate_msd(self, duration):
        condensed_MSD = np.zeros([duration, 2])
        dilute_MSD = np.zeros([duration, 2])
        condensed_MSD_std = np.zeros([duration, 2])
        dilute_MSD_std = np.zeros([duration, 2])
        amprec_extend = np.zeros([duration, self.n_ampar, 5])
        psdrec_extend = np.zeros([duration, self.n_psd95, 5])
        nmdrec_extend = np.zeros([duration, self.n_nmdar, 5])
        camrec_extend = np.zeros([duration, self.n_camk2, 5]) 
        condensed_particles = []
        dilute_particles = []
        for time in tqdm(range(0, duration)):
            self.calc_dilute_withxyz(time)
            _, _, _, amprec_extend[time,:,:], psdrec_extend[time,:,:], nmdrec_extend[time,:,:], camrec_extend[time,:,:] = self.variables
        amprec_extend[:,:,0:4], psdrec_extend[:,:,0:4], nmdrec_extend[:,:,0:4], camrec_extend[:,:,0:4] = self.convert_trajectory(duration)
        for p in range(0, self.n_ampar):
            if len(np.unique(amprec_extend[:,p,4])) > 1:
                print("remove No."+str(p))
            elif np.unique(amprec_extend[:,p,4])[0] == 0:
                dilute_particles.append(p)
            else:
                condensed_particles.append(p)
        for t in tqdm(range(1, duration)):  # duration
            condensed_MSD_meantemp = []
            dilute_MSD_meantemp = []
            for k in range(0, duration-t): # starting point
                condensed_MSD_temp = []
                dilute_MSD_temp =[]
                for p in range(0, self.n_ampar):
                    if p in condensed_particles:
                        refpos = amprec_extend[k+t,p,1:4]
                        condensed_MSD_temp.append(np.sum((refpos-amprec_extend[k,p,1:4])**2))
                    elif p in dilute_particles:
                        refpos = amprec_extend[k+t,p,1:4]
                        dilute_MSD_temp.append(np.sum((refpos-amprec_extend[k,p,1:4])**2))
                    else:
                        pass
                if len(condensed_MSD_temp) > 0:
                    condensed_MSD_meantemp.append(np.mean(condensed_MSD_temp))
                if len(dilute_MSD_temp) > 0:
                    dilute_MSD_meantemp.append(np.mean(dilute_MSD_temp)) # average per particles
            condensed_MSD[t,:] = np.array([t, np.mean(condensed_MSD_meantemp)])
            dilute_MSD[t,:] = np.array([t, np.mean(dilute_MSD_meantemp)])
            condensed_MSD_std[t,:] = np.array([t, np.std(condensed_MSD_meantemp)/np.sqrt(len(condensed_MSD_meantemp))])
            dilute_MSD_std[t,:] = np.array([t, np.std(dilute_MSD_meantemp)/np.sqrt(len(dilute_MSD_meantemp))])
        np.savetxt("c_MSDave_AMPAR.txt", condensed_MSD)
        np.savetxt("d_MSDave_AMPAR.txt", dilute_MSD)
        np.savetxt("c_MSDstd_AMPAR.txt", condensed_MSD_std)
        np.savetxt("d_MSDstd_AMPAR.txt", dilute_MSD_std)
        return None

    def plot_msd(self):
        import matplotlib
        palette = ["#d43f3a", "#eea236", "#5cb85c", "#46b8da", "#357ebd", "#9632b8", "#b8b8b8"]
        ####### settings #######
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
        fig = plt.figure(figsize=[5,2])
        ax = fig.add_subplot(111)
        cmsd = np.loadtxt("c_MSDave_AMPAR.txt")[:,:]
        dmsd = np.loadtxt("d_MSDave_AMPAR.txt")[:,:]
        cmsdstd = np.loadtxt("c_MSDstd_AMPAR.txt")[:,:]
        dmsdstd = np.loadtxt("d_MSDstd_AMPAR.txt")[:,:]
        ax.plot(cmsd[:,0]*8000*0.25/1e3, cmsd[:,1], color=palette[0])
        ax.plot(dmsd[:,0]*8000*0.25/1e3, dmsd[:,1], color=palette[2])
        # if necessary
        #for p in range(1, 5):
        #    cmsd = np.loadtxt("c_MSDave_AMPAR.txt")[:,:]
        #    dmsd = np.loadtxt("d_MSDave_AMPAR.txt")[:,:]
        #    ax.plot(cmsd[:,0]*4000*0.25/1e3, cmsd[:,1], color=palette[0], alpha=0.2*p)
        #    ax.plot(dmsd[:,0]*4000*0.25/1e3, dmsd[:,1], color=palette[2], alpha=0.2*p)
        #ax.errorbar(cmsdstd[:,0]*4000*0.25/1e3, cmsd[:,1], yerr=cmsdstd[:,1], color=palette[0], alpha=0.5, linewidth=0.5)
        #ax.errorbar(dmsdstd[:,0]*4000*0.25/1e3, dmsd[:,1], yerr=dmsdstd[:,1], color=palette[2], alpha=0.5, linewidth=0.5)
        ax.set_xlabel(r"$\Delta$ Time (μs)", fontsize=11)
        ax.set_xlim(0,100)
        ax.set_ylim(0,3000)
        ax.set_ylabel(r"MSD $(nm^2)$", fontsize=11)
        plt.savefig("MSD_trans.svg")
        plt.show()
        return None


#result = h5readMSD("strictampar_cutoff1.6_2.h5")
#print(result.n_ampar)
#result.calc_dilute_withxyz(5000)
#amprec, psdrec, nmdrec, camrec = result.convert_trajectory(3)
#print(amprec.shape, camrec.shape)
#result.track_trajectory(50,5)
#_, _, _, amp, _, _, _ = result.variables
#print(amp)
#result.calculate_msd(50)
#result.plot_msd()
