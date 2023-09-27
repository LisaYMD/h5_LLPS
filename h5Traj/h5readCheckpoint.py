import os
import sys
import numpy as np
import itertools
import readdy
from pint import UnitRegistry
import h5py
from .h5readTraj import h5readTraj
from tqdm import tqdm

# calculation of radius of each globular protein/domain
def size(aa):
    radius = 0.1 * 2.24 * (aa ** 0.392) # radius of globular protein(nanometer)
    return radius

class h5readCheckpoint(h5readTraj):
    def __init__(self, dir_name):
        self.dir_name = dir_name
        filename = dir_name + "/checkpoint_0.h5"
        with h5py.File(filename, 'r') as f:
            types = f['readdy']['config']['particle_types']
            self.types_str = [types[k][0].decode() for k in range(0, len(types))] 
            self.types_num = [types[k][1] for k in range(0, len(types))]
            info = np.atleast_1d(f['readdy']['config']['general'][...])[0].decode()
            traj = f['readdy']['trajectory']['trajectory_ckpt']
            #topo = f['readdy']['observables']['topologies_ckpt']
            limrec = traj['limits']
            rec = traj['records']
            ini = rec[limrec[0, 0]:limrec[0, 1]]
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

    def viewCheckpoint(self, number=-1):
        membrane_thickness = 5. # thickness of postsynaptic membrane is about 5 nm
        membrane_baseline = 0.  # assume that membrane is located at the center of box
        PSD_baseline = -50. # where psd 95 is anchored compared to the memebrane
        PSD_size = 100. # size of PSD is about 300 nm
        PSD_thickness = 50. # thickness of PSD is about 25-50 nm
        aa_ampar = 936
        #aa_ampar = 234 # monomer of DsRed (tetrameric FP: pseudo-ampar complex)
        aa_tarp = 211 # tarp
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
        #camk2_linker = 0. # inactivated CaMKII
        # system settings
        custom_unit = {"length_unit":"nanometer", "time_unit":"nanosecond", "energy_unit":"kilojoule/mol"}
        system = readdy.ReactionDiffusionSystem([self.xbox, self.xbox, self.zbox], temperature=300.*readdy.units.kelvin, periodic_boundary_conditions=[True, True, True], unit_system=custom_unit)
        ### PARTICLES
        # calculation of diffusion constant of each globular protein/domain
        def diff_const(size, string):
            rh = 1.45* size # radius of hydration (nanometer)
            # stokes-einstein expression
            RT = system.kbt.magnitude # kilojoule/mole = 10^(3)*Neuton*Meter/mole = 10^(3+9)*Neuton*nanometer/mole
            # Pascal = Neuton * (Meter)**(-2) = Neuton * nanometer^-2 * 10^(18)
            if string == "cell":
                eta = 0.85137 # milliPascal*second = 10^-3 * Neuton*nanometer^-2*10^(18)*nanosecond*10^9
            elif string == "lipid":
                eta = 0.85137*0.1 # milliPascal*second = 10^-3 * Neuton*nanometer^-2*10^(18)*nanosecond*10^9
            else:
                None
            diff = RT*10/(6.02214076*6*np.pi*eta*rh)
            return diff
        # types of particles
        system.add_topology_species("A", diffusion_constant=diff_const(size(aa_ampar), "cell")) # DsRed monomer
        system.add_topology_species("B0", diffusion_constant=diff_const(size(aa_tarp), "cell")) # TARP(unbinding)
        system.add_topology_species("B1", diffusion_constant=diff_const(size(aa_tarp), "cell")) # TARP(binding to PDZ)
        system.add_topology_species("C", diffusion_constant=diff_const(size(aa_ntd), "cell")) # NTD(palmitoylated)
        system.add_topology_species("D01", diffusion_constant=diff_const(size(aa_pdz), "cell")) # PDZ1 w/o PDB
        system.add_topology_species("D02", diffusion_constant=diff_const(size(aa_pdz), "cell")) # PDZ2 w/o PDB
        system.add_topology_species("D03", diffusion_constant=diff_const(size(aa_pdz), "cell")) # PDZ3 w/o PDB
        system.add_topology_species("D11", diffusion_constant=diff_const(size(aa_pdz), "cell")) # PDZ1 w/ PDB
        system.add_topology_species("D12", diffusion_constant=diff_const(size(aa_pdz), "cell")) # PDZ2 w/ PDB
        system.add_topology_species("D13", diffusion_constant=diff_const(size(aa_pdz), "cell")) # PDZ3 w/ PDB
        system.add_topology_species("E", diffusion_constant=diff_const(size(aa_sh3), "cell")) # SH3
        system.add_topology_species("F", diffusion_constant=diff_const(size(aa_gk), "cell")) # GK
        system.add_topology_species("G", diffusion_constant=diff_const(size(aa_nmdar), "cell")) # eqFP670 monomer
        system.add_topology_species("N00", diffusion_constant=diff_const(size(aa_glun2bc), "cell")) # GluN2Bc(unbinding)
        system.add_topology_species("N01", diffusion_constant=diff_const(size(aa_glun2bc), "cell")) # GluN2Bc(w/o PDZ, w/ kinase)
        system.add_topology_species("N10", diffusion_constant=diff_const(size(aa_glun2bc), "cell")) # GluN2Bc(w/ PDZ, w/o kinase)
        system.add_topology_species("N11", diffusion_constant=diff_const(size(aa_glun2bc), "cell")) # GluN2Bc(w/ PDZ, w/ kinase)
        system.add_topology_species("M", diffusion_constant=diff_const(camk2_hub, "cell")) # CaMKII hub complex
        system.add_topology_species("K0", diffusion_constant=diff_const(camk2_kinase, "cell")) # CaMKII kinase domain(unbinding)
        system.add_topology_species("K1", diffusion_constant=diff_const(camk2_kinase, "cell")) # CaMKII kinase domain(binding to GluN2Bc)
        ### TOPOLOGIES
        system.topologies.add_type("PSD")
        ### POTENTIALS
        # AMPAR-TARP complex
        system.topologies.configure_harmonic_bond("A", "B0", force_constant=10., length=(size(aa_ampar)+size(aa_tarp)))
        system.topologies.configure_harmonic_bond("A", "B1", force_constant=10., length=(size(aa_ampar)+size(aa_tarp)))
        system.topologies.configure_harmonic_bond("B0", "B0", force_constant=10., length=(size(aa_tarp)+size(aa_ampar))*np.sqrt(2))
        system.topologies.configure_harmonic_bond("B1", "B0", force_constant=10., length=(size(aa_tarp)+size(aa_ampar))*np.sqrt(2))
        system.topologies.configure_harmonic_bond("B1", "B1", force_constant=10., length=(size(aa_tarp)+size(aa_ampar))*np.sqrt(2))
        # NMDA receptor
        system.topologies.configure_harmonic_bond("G", "N00", force_constant=10., length=(size(aa_nmdar)+size(aa_glun2bc)))
        system.topologies.configure_harmonic_bond("G", "N01", force_constant=10., length=(size(aa_nmdar)+size(aa_glun2bc)))
        system.topologies.configure_harmonic_bond("G", "N10", force_constant=10., length=(size(aa_nmdar)+size(aa_glun2bc)))
        system.topologies.configure_harmonic_bond("G", "N11", force_constant=10., length=(size(aa_nmdar)+size(aa_glun2bc)))
        # PSD-95 # spring coefficient is determined by the number of amino acids 
        kbt = system.kbt.magnitude
        linker = 0.38
        link01 = 1.5*kbt/(60*linker**2)
        link12 = 1.5*kbt/(9*linker**2)
        link23 = 1.5*kbt/(67*linker**2)
        link3e = 1.5*kbt/(35*linker**2)
        linkef = 1.5*kbt/(36*linker**2)
        system.topologies.configure_harmonic_bond("C", "D01", force_constant=link01, length=(size(aa_ntd)+size(aa_pdz)))
        system.topologies.configure_harmonic_bond("C", "D11", force_constant=link01, length=(size(aa_ntd)+size(aa_pdz)))
        system.topologies.configure_harmonic_bond("D01", "D02", force_constant=link12, length=(size(aa_pdz)*2))
        system.topologies.configure_harmonic_bond("D01", "D12", force_constant=link12, length=(size(aa_pdz)*2))
        system.topologies.configure_harmonic_bond("D11", "D02", force_constant=link12, length=(size(aa_pdz)*2))
        system.topologies.configure_harmonic_bond("D11", "D12", force_constant=link12, length=(size(aa_pdz)*2))
        system.topologies.configure_harmonic_bond("D02", "D03", force_constant=link23, length=(size(aa_pdz)*2))
        system.topologies.configure_harmonic_bond("D02", "D13", force_constant=link23, length=(size(aa_pdz)*2))
        system.topologies.configure_harmonic_bond("D12", "D03", force_constant=link23, length=(size(aa_pdz)*2))
        system.topologies.configure_harmonic_bond("D12", "D13", force_constant=link23, length=(size(aa_pdz)*2))
        system.topologies.configure_harmonic_bond("D03", "E", force_constant=link3e, length=(size(aa_pdz)+size(aa_sh3)))
        system.topologies.configure_harmonic_bond("D13", "E", force_constant=link3e, length=(size(aa_pdz)+size(aa_sh3)))
        system.topologies.configure_harmonic_bond("E", "F", force_constant=linkef, length=(size(aa_sh3)+size(aa_gk)))
        d_list = ["D11", "D12", "D13"]
        for d in d_list:
            system.topologies.configure_harmonic_bond("B1", d, force_constant=10., length=(size(aa_tarp)+size(aa_pdz)))
            system.topologies.configure_harmonic_bond("N10", d, force_constant=10., length=(size(aa_glun2bc)+size(aa_pdz)))
            system.topologies.configure_harmonic_bond("N11", d, force_constant=10., length=(size(aa_glun2bc)+size(aa_pdz)))
        # CaMKII (I guess these potentials are not ideal for current situation)
        system.topologies.configure_harmonic_bond("M", "K0", force_constant=10., length=(camk2_hub+camk2_kinase+camk2_linker))
        system.topologies.configure_harmonic_bond("M", "K1", force_constant=10., length=(camk2_hub+camk2_kinase+camk2_linker))
        system.topologies.configure_harmonic_bond("K1", "N01", force_constant=10., length=(camk2_kinase+size(aa_glun2bc)))
        system.topologies.configure_harmonic_bond("K1", "N11", force_constant=10., length=(camk2_kinase+size(aa_glun2bc)))
        khdist = camk2_hub + camk2_kinase + camk2_linker
        height = 5.9/2 # cited from: Buonarati et al. (2021) cell reports
        #kkdist = khdist * 0.85   ### control the shape by regulating this parameter
        kkdist = np.sqrt(khdist**2- height**2) 
        kkh_angle = np.arccos(kkdist/(2*khdist))
        hkkk_angle = np.arccos(np.sqrt(3)*kkdist/(2*khdist*np.sin(kkh_angle)))
        #print(khdist*2, height, kkdist)
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
            number = latest
        else:
            simulation.load_particles_from_checkpoint(self.dir_name+"/checkpoint_"+str(number)+".h5")
        simulation.output_file = "ckpt_"+str(number)+".h5" 
        simulation.run(n_steps=0, timestep=0.25)
        simulation._show_progress = False
        radii_dict = {}
        color_dict = {}
        for r in range(0, len(self.types_str)):
            radii_dict[self.types_str[r]] = self.size_list[r]
            if self.types_str[r] == "A":
                color_dict[self.types_str[r]] = 1
            elif self.types_str[r][0] == "B":  ## several
                color_dict[self.types_str[r]] = 8
            elif self.types_str[r] == "C":
                color_dict[self.types_str[r]] = 10
            elif self.types_str[r][0] == "D":  ## several
                color_dict[self.types_str[r]] = 3
            elif self.types_str[r] == "E":
                color_dict[self.types_str[r]] = 7
            elif self.types_str[r] == "F":
                color_dict[self.types_str[r]] = 0
            elif self.types_str[r] == "G":
                color_dict[self.types_str[r]] = 4
            elif self.types_str[r][0] == "N":  ## several
                color_dict[self.types_str[r]] = 9
            elif self.types_str[r] == "M":
                color_dict[self.types_str[r]] = 2
            elif self.types_str[r][0] == "K":  ## several
                color_dict[self.types_str[r]] = 5
            else:
                print("ERROR: Unclassified particle type")
        ckpt = readdy.Trajectory(simulation.output_file)
        ckpt.convert_to_xyz(particle_radii=radii_dict, color_ids=color_dict, draw_box=True) 
        return None

    def checkpointToTraj(self):
        ckpt_list = sorted(os.listdir(self.dir_name))
        point1 = int(ckpt_list[1][11:-3]) ## strictly speaking, this is not correct
        listsum = 0
        maximum = {}
        maximum_trajectory = []
        for l in range(0, len(ckpt_list)):
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
            for l in range(0, len(ckpt_list)):
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
                                outfile.writelines(mk + " 0 0 0\n")
                        else:
                            pass
                    # if not, fill them by zero
                    else:
                        molcount = maximum[mk]
                        for d in range(0, molcount):
                            outfile.write(mk + " 0 0 0\n")
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

