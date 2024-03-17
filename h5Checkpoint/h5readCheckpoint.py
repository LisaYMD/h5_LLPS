import os
import sys
import numpy as np
import itertools
import readdy
from pint import UnitRegistry
import h5py
from tqdm import tqdm
import json

############### CAUTION:BEFORE USE ###############

# install hdf5plugin, readdy, h5py before use
# activate readdy environment by conda:
#     source activate readdy

##################################################

class h5readCheckpoint:
    def __init__(self, dir_name, dim=3):
        self.dir_name = dir_name
        filename = dir_name + "/checkpoint_0.h5"
        with h5py.File(filename, 'r') as f:
            types = f['readdy']['config']['particle_types']
            self.types_str = [types[k][0].decode() for k in range(0, len(types))] 
            self.types_num = [types[k][1] for k in range(0, len(types))]
            self.diff_const = [types[k][2] for k in range(0, len(types))]
            info = np.atleast_1d(f['readdy']['config']['general'][...])[0].decode()
            traj = f['readdy']['trajectory']['trajectory_ckpt']
            topo = f['readdy']['observables']['topologies_ckpt']
            limrec = traj['limits']
            rec = traj['records']
            ini = rec[limrec[0, 0]:limrec[0, 1]]
        boxinfo = info[(info.find('"box_size":[')+len(str('"box_size":['))):(info.find("],"))].split(",")
        self.xbox = np.array(boxinfo).astype("float")[0]
        self.zbox = np.array(boxinfo).astype("float")[2]
        self.size_list = []
        self.color_list = []
        # load setting file here
        dirname = os.path.dirname(__file__)
        if dim == 3:
            filename = os.path.join(dirname, "../mol_settings/3d_settings.json")
            with open(filename) as f:
                settings = json.load(f)
        elif dim == 2:
            filename = os.path.join(dirname, "../mol_settings/2d_settings.json")
            with open(filename) as f:
                settings = json.load(f)
        else:
            print("Unexpected dimension type")
        self.mlists =  settings["molecule"]
        for l in self.types_str:
            for m in range(0, len(self.mlists)):
                cand = settings["particle_type"][self.mlists[m]]
                if l in cand:
                    self.size_list.append(settings["size"][self.mlists[m]][cand.index(l)])
                    self.color_list.append(settings["color"][self.mlists[m]][cand.index(l)])
                    break
        assert len(self.types_str) == len(self.size_list)
        self.molcounts = np.zeros(len(self.mlists))
        for mol in range(0, len(self.mlists)):
            char = settings["particle_type"][self.mlists[mol]][0]
            if char in self.types_str:
                self.molcounts[mol]=len([ini[r] for r in range(0, len(ini)) if (ini[r][0]==self.types_num[self.types_str.index(char)])]) 

    def viewCheckpoint(self, number=-1):
        # system settings
        custom_unit = {"length_unit":"nanometer", "time_unit":"nanosecond", "energy_unit":"kilojoule/mol"}
        system = readdy.ReactionDiffusionSystem([self.xbox, self.xbox, self.zbox], temperature=300.*readdy.units.kelvin, periodic_boundary_conditions=[True, True, True], unit_system=custom_unit)
        ### PARTICLES
        # types of particles
        for l in range(0, len(self.types_str)):
            system.add_topology_species(self.types_str[l], diffusion_constant=self.diff_const[l])
        ### TOPOLOGIES
        system.topologies.add_type("PSD")
        ### POTENTIALS
        # AMPAR-TARP complex
        system.topologies.configure_harmonic_bond("A", "B0", force_constant=10., length=1.)
        system.topologies.configure_harmonic_bond("A", "B1", force_constant=10., length=1.)
        system.topologies.configure_harmonic_bond("B0", "B0", force_constant=10., length=1.)
        system.topologies.configure_harmonic_bond("B1", "B0", force_constant=10., length=1.)
        system.topologies.configure_harmonic_bond("B1", "B1", force_constant=10., length=1.)
        # NMDA receptor
        system.topologies.configure_harmonic_bond("G", "N00", force_constant=10., length=1.)
        system.topologies.configure_harmonic_bond("G", "N01", force_constant=10., length=1.)
        system.topologies.configure_harmonic_bond("G", "N10", force_constant=10., length=1.)
        system.topologies.configure_harmonic_bond("G", "N11", force_constant=10., length=1.)
        # PSD-95 # spring coefficient is determined by the number of amino acids 
        kbt = system.kbt.magnitude
        linker = 0.38
        link01 = 1.5*kbt/(60*linker**2)
        link12 = 1.5*kbt/(9*linker**2)
        link23 = 1.5*kbt/(67*linker**2)
        link3e = 1.5*kbt/(35*linker**2)
        linkef = 1.5*kbt/(36*linker**2)
        system.topologies.configure_harmonic_bond("C", "D01", force_constant=link01, length=1.)
        system.topologies.configure_harmonic_bond("C", "D11", force_constant=link01, length=1.)
        system.topologies.configure_harmonic_bond("D01", "D02", force_constant=link12, length=1.)
        system.topologies.configure_harmonic_bond("D01", "D12", force_constant=link12, length=1.)
        system.topologies.configure_harmonic_bond("D11", "D02", force_constant=link12, length=1.)
        system.topologies.configure_harmonic_bond("D11", "D12", force_constant=link12, length=1.)
        system.topologies.configure_harmonic_bond("D02", "D03", force_constant=link23, length=1.)
        system.topologies.configure_harmonic_bond("D02", "D13", force_constant=link23, length=1.)
        system.topologies.configure_harmonic_bond("D12", "D03", force_constant=link23, length=1.)
        system.topologies.configure_harmonic_bond("D12", "D13", force_constant=link23, length=1.)
        system.topologies.configure_harmonic_bond("D03", "E", force_constant=link3e, length=1.)
        system.topologies.configure_harmonic_bond("D13", "E", force_constant=link3e, length=1.)
        system.topologies.configure_harmonic_bond("E", "F", force_constant=linkef, length=1.)
        d_list = ["D11", "D12", "D13"]
        for d in d_list:
            system.topologies.configure_harmonic_bond("B1", d, force_constant=10., length=1.)
            system.topologies.configure_harmonic_bond("N10", d, force_constant=10., length=1.)
            system.topologies.configure_harmonic_bond("N11", d, force_constant=10., length=1.)
        # CaMKII (I guess these potentials are not ideal for current situation)
        system.topologies.configure_harmonic_bond("M", "K0", force_constant=10., length=1.)
        system.topologies.configure_harmonic_bond("M", "K1", force_constant=10., length=1.)
        system.topologies.configure_harmonic_bond("K1", "N01", force_constant=10., length=1.)
        system.topologies.configure_harmonic_bond("K1", "N11", force_constant=10., length=1.)
        khdist = 10
        height = 5.9/2 # cited from: Buonarati et al. (2021) cell reports
        #kkdist = khdist * 0.85   ### control the shape by regulating this parameter
        kkdist = np.sqrt(khdist**2- height**2) 
        # bond potential (hub-kinase)
        system.topologies.configure_harmonic_bond("M", "K0", force_constant=10., length=khdist)
        system.topologies.configure_harmonic_bond("M", "K1", force_constant=10., length=khdist)
        # bond potential (kinase-kinase)
        system.topologies.configure_harmonic_bond("K0", "K0", force_constant=10., length=kkdist)
        system.topologies.configure_harmonic_bond("K1", "K1", force_constant=10., length=kkdist)
        system.topologies.configure_harmonic_bond("K1", "K0", force_constant=10., length=kkdist)
        # simulation settings
        simulation = system.simulation(kernel="CPU")
        simulation.reaction_handler = "Gillespie"
        # run simulation
        simulation.observe.topologies(1)
        simulation.record_trajectory(1)
        if number < 0:
            # Default: read latest checkpoint
            simulation.load_particles_from_latest_checkpoint(self.dir_name+"/")
            number = 1
        else:
            simulation.load_particles_from_checkpoint(self.dir_name+"/checkpoint_"+str(number)+".h5")
        simulation.output_file = "ckpt_"+str(number)+".h5" 
        simulation.run(n_steps=0, timestep=0.25)
        simulation._show_progress = False
        radii_dict = {}
        color_dict = {}
        for r in range(0, len(self.types_str)):
            radii_dict[self.types_str[r]] = self.size_list[r]
            color_dict[self.types_str[r]] = self.color_list[r]
        ckpt = readdy.Trajectory(simulation.output_file)
        ckpt.convert_to_xyz(particle_radii=radii_dict, color_ids=color_dict, draw_box=True) 
        return None

    def checkpointToTraj(self):
        ckpt_list = sorted(os.listdir(self.dir_name))
        num_list = []
        for k in range(0, len(ckpt_list)):
            num_list.append(int(ckpt_list[k][11:-3]))
        ckpt_num_list = sorted(num_list)
        point1 = int(ckpt_num_list[1])
        listsum = 0
        maximum = {}
        maximum_trajectory = []
        for l in range(0, len(ckpt_num_list)):
            fname = "ckpt_"+str(listsum)+".h5"
            self.viewCheckpoint(listsum)
            with open(fname+".xyz", "r") as infile:
                xyz = infile.readlines()
                new_maximum = {} # {"typename": number of molecules}
                typecount = 0
                for i in range(0, len(xyz)):
                    if i == 0 or i == 1:
                        pass
                    else:
                        typename = xyz[i].split()[0]
                        if i == 2:
                            pass
                        elif i == len(xyz)-1:
                            typecount += 2
                            new_maximum[typename] = typecount
                        elif typename == oldtypename:
                            typecount += 1
                            oldtypename = typename
                        elif typename != oldtypename:
                            typecount += 1
                            new_maximum[oldtypename] = typecount
                            typecount = 0
                        else:
                            print("error")
                        oldtypename = typename
            if l == 0:
                maximum = new_maximum.copy()
            else:
                for key, value in new_maximum.items():
                    if key in maximum.keys():
                        if maximum[key] < value:
                            maximum[key] = value
                    else:
                        maximum[key] = value
            sub_maximum = new_maximum.copy()
            maximum_trajectory.append(sub_maximum)
            listsum += point1
        #maximum_sort = sorted(maximum_trajectory, key=lambda x:x["time"])
        listsum = 0
        valuesum = sum(maximum.values()) 
        # write the stored maximum value
        with open(str(self.dir_name)+".xyz", "w") as outfile:
            for l in range(0, len(ckpt_num_list)):
                fname = "ckpt_"+str(listsum)+".h5"
                with open(fname+".xyz", "r") as infile2:
                    xyz = infile2.readlines()
                outfile.write(str(valuesum)+"\n") # firstline: number of molecules
                outfile.write(xyz[1]) # secondline : indent
                molindex = [0,] + list(itertools.accumulate(maximum_trajectory[l].values()))
                moltypes = list(maximum_trajectory[l].keys())
                for mk in maximum.keys():
                    if mk in maximum_trajectory[l].keys():
                        # if the type exists, write current positions
                        molcount = maximum_trajectory[l][mk]
                        start = molindex[moltypes.index(mk)]+2
                        outfile.writelines(xyz[start:start+molcount])
                        start = 0
                        # if the time does not reach maximum value, fill them by zero
                        if molcount < maximum[mk]:
                            deficient = int(maximum[mk]-molcount)
                            for d in range(0, deficient):
                                outfile.writelines(mk+" "+str(self.zbox)+" "+str(self.zbox)+" "+str(self.zbox)+"\n") #" 0 0 0\n")
                        else:
                            pass
                    # if not, fill them by zero
                    else:
                        molcount = maximum[mk]
                        for d in range(0, molcount):
                            outfile.write(mk+" "+str(self.zbox)+" "+str(self.zbox)+" "+str(self.zbox)+"\n") #(mk + " 0 0 0\n")
                listsum += point1
                ## rather than erasing the xyz/tcl file, calculate the number of each particle types and store them
                os.remove(fname)
                os.remove(fname+".xyz")
                if l != 0:
                    os.remove(fname+".xyz.tcl")
        with open("ckpt_0.h5.xyz.tcl", "r") as tcl:
            lines = tcl.readlines()
        lines[1] = "mol load xyz "+str(self.dir_name)+".xyz\n"
        with open(str(self.dir_name)+".xyz.tcl", "w") as outtcl:
            outtcl.writelines(lines)
        os.remove("ckpt_0.h5.xyz.tcl") 
        return None

