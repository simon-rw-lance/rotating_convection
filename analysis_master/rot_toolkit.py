from dedalus import public as de
from dedalus.extras import flow_tools
import h5py
import numpy as np
import matplotlib.pyplot as plt
import sys
import pathlib
import time
import datetime
from scipy import integrate
from scipy.signal import savgol_filter


class SimData:
    def __init__(self, data_direc=None, save_direc=None, name=None, id='', making_figs=False):

        self.test = "Yes"

        timestamp = datetime.datetime.now().strftime("%m-%dT%H-%M-%S")

        self.name = name
        self.data_direc = data_direc
        self.save_direc = save_direc
        self.id = f"_{id}"

        self.snapshot_data_read = False
        self.analysis_data_read = False
        self.avg_index_found = False

        if making_figs:
            pathlib.Path(self.save_direc).mkdir(parents=True, exist_ok=True)

        print("Accessing simulation: {}".format(self.name))

        # Space and time arrays are used in a lot of methods and so loaded on initalisation
        with h5py.File(f"{self.data_direc}snapshots/snapshots{self.id}.h5", mode='r') as file:
            self.snap_t = np.array(file['scales']['sim_time'])
            self.snap_length = len(self.snap_t)
            self.z = np.array(file['scales']['z']['1.0'])
        with h5py.File(f"{self.data_direc}analysis/analysis{self.id}.h5", mode='r') as file:
            self.ana_t = np.array(file['scales']['sim_time'])
            self.ana_length = len(self.ana_t)

        # Read in run_parameters file and store them in a dictionary called 'params'
        with h5py.File(f"{data_direc}run_parameters/run_parameters{self.id}.h5", mode='r') as file:
            vars = list(file['tasks'])
            params = {}
            for var in vars:
                vs = np.shape(file['tasks'][var])
                if vs == (1,1,1):
                    params[var] = file['tasks'][var][0,0,0]
                elif vs[2] != 1:
                    params[var] = file['tasks'][var][0,0,:]
                elif vs[1] != 1:
                    params[var] = file['tasks'][var][0,:,0]
                elif vs[0] != 1:
                    params[var] = file['tasks'][var][:,0,0]
        self.params = params

    ##################################
    ##### Data Reading Functions #####
    ##################################

    def DataRead(self,var,file,tree='tasks'):
        '''
        Read in a specific variable from a given datafile.
        Note: Variable stored externally and not accessible via self.var while
        inside the class. To recrate a class variable in another method, call
        this function there,
        e.g. self.var = self.DataRead(var,file,tree)

        var = variable to be read in
        file = which datafile (snapshots, analysis, run_parameters etc.)
        tree = 'scales' or 'tasks'
        '''
        with h5py.File(f"{self.data_direc}{file}/{file}{id}.h5", mode='r') as file:
            x=np.array(file[tree][var])
        return(x)

    def ReadSnapshots(self):
        '''
        Function that reads in all variable stored in the 'snapshots' .h5 file
        and stores them in two dictionaries, snap_scales and snap_tasks
        '''
        print("### Loading Snapshot Data... ###")

        self.snap_scales = {}
        self.snap_tasks = {}

        with h5py.File(f"{self.data_direc}snapshots/snapshots{self.id}.h5", mode='r') as file:
            scales = list(file['scales'])
            tasks = list(file['tasks'])
            for var in scales:
                if var not in ('x', 'y', 'z'):
                    self.snap_scales[var] = np.array(file['scales'][var])
                else:
                    if len(list(file['scales'][var])):
                        self.snap_scales[var] = np.array(file['scales'][var]['1.0'])
                    elif var in ('x', 'y'): # If no horizontal domain found, assume constant grid spacing
                        self.ana_scales[var] = np.linspace(0, self.params[f'L{var}'], int(self.params[f'N{var}']))
            for var in tasks:
                self.snap_tasks[var] = np.array(file['tasks'][var])

        theta = 1-np.exp(-self.params['Np']/self.params['m'])
        rho_ref = (1-theta*self.z)**(self.params['m'])
        s_base = 1/(theta*self.params['m']) * ( (1-theta)**(-self.params['m']) - 1/rho_ref )
        self.params['theta'] = theta
        self.params['rho_ref'] = rho_ref
        self.snap_tasks['s_base'] = s_base

        s_tot = np.zeros_like(self.snap_tasks['s'])
        for i in range(len(self.snap_tasks['s'][:,0,0])):
            for j in range(len(self.snap_tasks['s'][0,:,0])):
                s_tot[i,j,:] = self.snap_tasks['s'][i,j,:] + s_base
        self.snap_tasks['s_tot'] = s_tot

        self.snap_t = self.snap_scales['sim_time']
        self.snapshot_data_read = True


        print("### Snapshots Loaded ###")

    def ReadAnalysis(self):
        '''
        Function that reads in all variable stored in the 'analysis' .h5 file
        and stores them in two dictionaries, ana_scales and ana_tasks
        '''

        self.ana_scales = {}
        self.ana_tasks = {}

        with h5py.File(f"{self.data_direc}analysis/analysis{self.id}.h5", mode='r') as file:
            scales = list(file['scales'])
            tasks = list(file['tasks'])
            for var in scales:
                if var not in ('x', 'y', 'z'):
                    self.ana_scales[var] = np.array(file['scales'][var])
                else:
                    if len(list(file['scales'][var])):
                        self.ana_scales[var] = np.array(file['scales'][var]['1.0'])
                    elif var in ('x', 'y'): # If no horizontal domain found, assume constant grid spacing
                        self.ana_scales[var] = np.linspace(0, self.params[f'L{var}'], int(self.params[f'N{var}']))
            for var in tasks:
                self.ana_tasks[var] = np.array(file['tasks'][var])

        self.ana_t = self.ana_scales['sim_time']
        self.analysis_data_read = True

    ##################################
    ######### Calculus Tools #########
    ##################################

    # These need editing and OOP-ing

    def t_int(self, f,t,T=0):
        '''
        Take time average of 2d spatial data
        '''
        T = t[-1] - t[0]  #Default T set to entire input t array
        f_t = np.zeros_like(f[0,:,:])
        for i in range(len(f[0,:,0])):
            for j in range(len(f[0,0,:])):
                f_t[i,j] = integrate.cumtrapz(f[:,i,j], t)[-1]/T
        return(f_t)

    def FD_fourth_per(self, f, del_x):
        fx=np.zeros_like(f)
        ly = len(f[:,0])
        for i in range(ly): #Loop over x
            for j in range(len(f[0,:])): #Loop over z

                i1, i2, i3, i4 = i-2, i-1, i+1, i+2
                if i == 0:
                    i1, i2 = i-2+ly, i-1+ly
                elif i == 1:
                    i1 = i-2+ly
                elif i == ly-1:
                    i3, i4 = i+1-ly, i+2-ly
                elif i == ly-2:
                    i4 = i+2-ly
                fx[i,j] = ( (1/2)*f[i1,j] + (-2/3)*f[i2,j] + (2/3)*f[i3,j] + (-1/12)*f[i4,j] ) / del_x
        return(fx)

    ##################################
    ######### Analysis Tools #########
    ##################################

    def RuntimeCheck(self):
        '''
        Checks the runtime of a simulation.
        Prints both the in-simulation and wall time to terminal.
        '''

        with h5py.File(f"{self.data_direc}analysis/analysis{self.id}.h5", mode='r') as file:
            wall_t = np.array(file['scales']['world_time'])
            print(f"Simulation length = {self.ana_t[-1]} tau_nu")
            print(f"Runtime = {(wall_t[-1]-wall_t[0])/(60*60)} hours")

    def AvgIndex(self, start=0, stop=np.nan):
        '''
        Finds the array index values for both the analysis and snapshot data files
        from a given simulation time range.
        ASI and AEI are the start and end index values for analysis files
        SSI and SEI are the start and end index values for snapshot files
        '''

        print("### Finding indexs for time averaging ###")

        self.start = start
        self.stop = stop

        if (self.start==0 and np.isnan(self.stop)):
            print("WARNING: Average index set across the entire domain.")
        if (self.start < self.ana_t[0] or self.stop < self.ana_t[0]):
            sys.exit(f"Average time period out of simulation range: {self.ana_t[0]} -> {self.ana_t[-1]}")
        if self.start > self.ana_t[-1]:
            sys.exit(f"Average time period out of simulation range: {self.ana_t[0]} -> {self.ana_t[-1]}")
        if self.stop > self.ana_t[-1]:
            print("Stop time larger than total runtime. Set to end of simulation")
            self.stop = self.ana_t[-1]

        # Finding the index value in the t arrays for start and stop points
        self.ASI = (np.abs(self.ana_t  - start)).argmin()  # analysis start index
        self.SSI = (np.abs(self.snap_t - start)).argmin() # snapshot start index
        if np.isnan(stop): # End of array if NaN value given
            self.AEI, self.SEI = -1, -1
            self.stop = self.ana_t[-1]
        else:
            self.AEI = (np.abs(self.ana_t  - stop)).argmin()   # analysis end index
            self.SEI = (np.abs(self.snap_t - stop)).argmin()  # snapshot end index
        self.avg_t_range = self.ana_t[self.AEI] - self.ana_t[self.ASI]
        print(f"Average range set as: {self.avg_t_range:.3f}")

        self.avg_index_found = True

    ##################################
    ######### Plotting Tools #########
    ##################################

    def LimFinder(self,f):
        f_max = np.max(f)
        f_min = np.min(f)
        if f_max >= abs(f_min):
            lims = [-f_max, f_max]
        else:
            lims = [f_min, abs(f_min)]
        return(lims)

    def ContourPlot(self, f, x, z, colors, labels, fig=None, ax=None, lims=None, cont_res=51):
        '''
        Function for plotting a 2d field.
        Dependencies:
        - ReadSnapshots

        labels = [x_axis, y_axis, cbar]
        '''

        if (fig == None) or (ax == None):
            fig, ax = plt.subplots()

        if lims == None:
            lims = [np.min(f), np.max(f)]

        xx, zz = np.meshgrid(x, z)
        map = ax.contourf(xx, zz, f, cmap=colors, levels=np.linspace(lims[0],lims[1],cont_res))
        bar = fig.colorbar(map, ax=ax)
        ax.set_xlabel(labels[0])
        ax.set_ylabel(labels[1])
        bar.set_label(labels[2], rotation=0)

    def FieldEvo(self, labels, cadence=5):
        '''
        Create an series of plots showing the temporal evolution of a 2D field.
        '''

        print(f"### Producing evolution plots for {labels[2]}, cadence = {cadence} ###")

        pathlib.Path(f"{self.save_direc}{labels[2]}_evo").mkdir(parents=True, exist_ok=True)

        f = np.array(self.snap_tasks[labels[2]])

        if 'x' in self.snap_scales:
            x = self.snap_scales['x']
        elif 'y' in self.snap_scales:
            x = self.snap_scales['y']
        else:
            print("Error: No horizontal domain found.")
        z = self.snap_scales['z']

        # Set colorbar information based on field label.
        if labels[2] in ['u', 'v', 'w']:
            # If field is a velocity, find largest absolute value using LimFinder.
            lims = self.LimFinder(f)
            colors = "RdBu_r"
        elif labels[2] == 's':
            # If field is entropy, set minimum to 0, and max to largest value
            lims = [0, np.max(f)]
            colors = 'OrRd'
        else:
            # Unrecognised field, default to max/min values of field.
            lims = [np.min(f), np.max(f)]
            colors = "OrRd"

        counter = 1
        for i in range(0, len(f[:,0,0]), cadence):
            if int((i/len(f[:,0,0]))*10) == counter:
                counter += 1
                print(f"{int(100*i/len(f[:,0,0]))}% complete")
            fig = plt.figure(figsize=(6,4))
            ax = fig.add_subplot()
            self.ContourPlot(f[i,:,:].T, x, z, colors, labels, fig, ax, lims)
            plt.savefig(f"{self.save_direc}{labels[2]}_evo/{i:04d}")
            plt.clf()
            plt.close()

    def SnapshotEvo(self, cadence=5):
        '''
        Creates a series of plots showing the temporal evolution of s, u, v, w, and KE
        Dependencies:
        - ReadSnapshots
        - ReadAnalysis
        '''

        pathlib.Path(f"{self.save_direc}snapshot_evo").mkdir(parents=True, exist_ok=True)

        s_all = self.snap_tasks['s']
        u_all = self.snap_tasks['u']
        v_all = self.snap_tasks['v']
        w_all = self.snap_tasks['w']
        KE = self.ana_tasks['KE'][:,0,0]

        if 'x' in self.snap_scales:
            x = self.snap_scales['x']
        elif 'y' in self.snap_scales:
            x = self.snap_scales['y']
        else:
            print("Error: No horizontal domain found.")
        z = self.snap_scales['z']

        u_lims = self.LimFinder(u_all)
        v_lims = self.LimFinder(v_all)
        w_lims = self.LimFinder(w_all)
        s_lims = [0, np.max(s_all)]

        for i in range(0,self.snap_length,cadence):

            ana_index = (np.abs(self.ana_t - self.snap_t[i])).argmin()

            fig = plt.figure(figsize=(16,12))
            gs = fig.add_gridspec(3,2, bottom=0.08, top=0.92, hspace=0.25, wspace=0.25)
            ax1 = fig.add_subplot(gs[0,0])
            ax2 = fig.add_subplot(gs[1,0])
            ax3 = fig.add_subplot(gs[0,1])
            ax4 = fig.add_subplot(gs[1,1])
            ax5 = fig.add_subplot(gs[2,:])
            self.ContourPlot(u_all[i,:,:].T, x, z, 'RdBu_r', ['y','z','u'], fig, ax1, u_lims)
            self.ContourPlot(v_all[i,:,:].T, x, z, 'RdBu_r', ['y','z','v'], fig, ax2, v_lims)
            self.ContourPlot(w_all[i,:,:].T, x, z, 'RdBu_r', ['y','z','w'], fig, ax3, w_lims)
            self.ContourPlot(s_all[i,:,:].T, x, z, 'OrRd',   ['y','z','s'], fig, ax4, s_lims)

            ax5.set_xlim(self.ana_t[0], self.ana_t[-1])
            ax5.set_ylim(0,1.1*np.max(KE))
            ax5.plot(self.ana_t[0:ana_index], KE[0:ana_index], label='KE')
            ax5.set_xlabel(r"Time / $\tau_\nu$")
            ax5.set_ylabel("KE")
            ax5.legend()
            plt.savefig(f"{self.save_direc}snapshot_evo/{i:04d}")
            plt.clf()
            plt.close()



    def PlotSnapshots_xyz(self):
        '''
        Produces a time averaged plot of all the velocities and the entropy field.
        Dependencies:
        - ReadSnapshots
        - ReadAnalysis
        - AvgIndex
        '''

        s = np.mean(np.array(self.snap_tasks['s'])[self.SSI:self.SEI,:,:], axis=0)
        u = np.mean(np.array(self.snap_tasks['u'])[self.SSI:self.SEI,:,:], axis=0)
        v = np.mean(np.array(self.snap_tasks['v'])[self.SSI:self.SEI,:,:], axis=0)
        w = np.mean(np.array(self.snap_tasks['w'])[self.SSI:self.SEI,:,:], axis=0)
        KE = self.ana_tasks['KE'][:,0,0]

        fig = plt.figure(figsize=(16,12))
        gs = fig.add_gridspec(3,2, bottom=0.08, top=0.92, hspace=0.25, wspace=0.25)
        ax1 = fig.add_subplot(gs[0,0])
        ax2 = fig.add_subplot(gs[1,0])
        ax3 = fig.add_subplot(gs[0,1])
        ax4 = fig.add_subplot(gs[1,1])
        ax5 = fig.add_subplot(gs[2,:])

        if 'x' in self.snap_scales:
            x = self.snap_scales['x']
        elif 'y' in self.snap_scales:
            x = self.snap_scales['y']
        else:
            print("Error: No horizontal domain found.")
        z = self.snap_scales['z']

        u_lims = self.LimFinder(u)
        v_lims = self.LimFinder(v)
        w_lims = self.LimFinder(w)
        s_lims = [0, np.max(s)]

        self.ContourPlot(u.T, x, z, 'RdBu_r', ['y','z','u'], fig, ax1, u_lims)
        self.ContourPlot(v.T, x, z, 'RdBu_r', ['y','z','v'], fig, ax2, v_lims)
        self.ContourPlot(w.T, x, z, 'RdBu_r', ['y','z','w'], fig, ax3, w_lims)
        self.ContourPlot(s.T, x, z, 'OrRd',   ['y','z','s'], fig, ax4, s_lims)

        ax5.set_xlim(self.ana_t[0], self.ana_t[-1])
        ax5.set_ylim(0,1.1*np.max(KE))
        ax5.plot(self.ana_t, KE, label='KE')
        ax5.set_xlabel(r"Time / $\tau_\nu$")
        ax5.set_ylabel("KE")
        ax5.plot([self.ana_t[self.ASI], self.ana_t[self.ASI]],[0, 1.1*np.max(KE)], 'r', linestyle='--', label='Time average range')
        ax5.plot([self.ana_t[self.AEI], self.ana_t[self.AEI]],[0, 1.1*np.max(KE)], 'r', linestyle='--')
        ax5.legend()
        plt.savefig(f"{self.save_direc}avg_state")
        plt.clf()
        plt.close()

    def PlotSProfile(self):
        '''
        Produces vertical profile of the horizontally averaged entropy.
        '''

        sx = np.mean(np.array(self.analysis_tasks['s_x'])[self.SSI:self.SEI,:,:], axis=0)

        if self.avg_index_found == False:
            print("Error: Couldn't plot entropy profile, average indexs not found.")
            return

        plt.plot(self.mean_sx, self.z)
        plt.xlabel("s")
        plt.ylabel("z")
        plt.savefig(self.save_direc + "s_profile")
        plt.clf()
        plt.close()

    def PlotKE(self):
        '''
        Produces a timeseries plot of the domain averaged kinetic energy. If averaging indexs
        have been found using the AvgIndex function, then the averaging range is also indicated.
        '''

        print("### Plotting KE ###")

        self.KE = self.ana_tasks['KE'][:,0,0]

        plt.figure(figsize=(12,4))
        KE_fig = plt.subplot(111)
        KE_box = KE_fig.get_position()
        KE_fig.set_position([KE_box.x0, KE_box.y0+(0.1*KE_box.height), KE_box.width, KE_box.height*0.8])
        plt.xlabel(r"Time $[\tau_{\nu}]$")
        plt.ylabel("KE")
        plt.xlim(self.ana_t[0],self.ana_t[-1])
        plt.ylim(0,1.1*np.max(self.KE))
        plt.plot(self.ana_t, self.KE,  'C0', label='Integral average - Dedalus')
        if self.avg_index_found == True:
            plt.plot([self.ana_t[self.ASI], self.ana_t[self.ASI]],[0, 1.1*np.max(self.KE)], 'r', linestyle='--', label='Time average range')
            plt.plot([self.ana_t[self.AEI], self.ana_t[self.AEI]],[0, 1.1*np.max(self.KE)], 'r', linestyle='--')
            plt.legend()
        plt.savefig(f"{self.save_direc}KE")
        plt.clf()
        plt.close()

    def FluxTotal(self):
        '''
        Dependencies:
        - ReadAnalysis
        - AvgIndex
        '''

        self.z = self.ana_scales['z']

        self.L_cond_all = self.ana_tasks['L_cond'][:,0,:]
        self.L_conv_all = self.ana_tasks['L_conv'][:,0,:]
        self.L_buoy_all = self.ana_tasks['L_buoy'][:,0,:]
        self.L_diss_all = self.ana_tasks['L_diss'][:,0,:]

        self.L_p_all = self.ana_tasks['L_p'][:,0,:]
        self.L_enth_all = self.ana_tasks['L_enth'][:,0,:]
        self.L_visc_all = self.ana_tasks['L_visc'][:,0,:]
        self.L_KE_all = self.ana_tasks['L_KE'][:,0,:]

        self.L_cond = np.mean(self.L_cond_all[self.ASI:self.AEI], axis=0)
        self.L_conv = np.mean(self.L_conv_all[self.ASI:self.AEI], axis=0)
        self.L_buoy = np.mean(self.L_buoy_all[self.ASI:self.AEI], axis=0)
        self.L_diss = np.mean(self.L_diss_all[self.ASI:self.AEI], axis=0)
        self.L_p    = np.mean(self.L_p_all[self.ASI:self.AEI],    axis=0)
        self.L_enth = np.mean(self.L_enth_all[self.ASI:self.AEI], axis=0)
        self.L_visc = np.mean(self.L_visc_all[self.ASI:self.AEI], axis=0)
        self.L_KE   = np.mean(self.L_KE_all[self.ASI:self.AEI],   axis=0)
        self.int_L_tot = self.L_cond + self.L_conv + self.L_buoy + self.L_diss
        self.tot_L_tot = self.L_cond + self.L_enth + self.L_visc + self.L_KE


    def FluxPlots(self):
        '''
        Dependencies:
        - ReadAnalysis
        - AvgIndex
        '''

        self.FluxTotal()
        print("### Plotting fluxes ###")

        plt.plot(self.L_cond, self.z, label=f"$L_{{cond}}$")
        plt.plot(self.L_conv, self.z, label=f"$L_{{conv}}$")
        plt.plot(self.L_buoy, self.z, label=f"$L_{{diss}}$")
        plt.plot(self.L_diss, self.z, label=f"$L_{{buoy}}$")
        plt.plot(self.int_L_tot, self.z, c='k')
        plt.title(f"$\Delta L_{{tot}} = {np.max(abs(self.int_L_tot-1)):.4f}$, Time average $= {self.avg_t_range:.3f} \\tau$")
        plt.ylabel("z")
        plt.xlabel(r"$L / L_{total}$")
        plt.legend()
        plt.savefig(f"{self.save_direc}intE_fluxes")
        plt.clf()
        plt.close()

        plt.plot(self.L_cond, self.z, label=f"$L_{{cond}}$")
        plt.plot(self.L_enth, self.z, label=f"$L_{{enth}}$")
        plt.plot(self.L_visc, self.z, label=f"$L_{{visc}}$")
        plt.plot(self.L_KE,   self.z, label=f"$L_{{KE}}$")
        plt.plot(self.tot_L_tot, self.z, c='k')
        plt.title(f"$\Delta L_{{tot}} = {np.max(abs(self.tot_L_tot-1)):.4f}$, Time average $= {self.avg_t_range:.3f} \\tau$")
        plt.ylabel("z")
        plt.xlabel(r"$L / L_{total}$")
        plt.legend()
        plt.savefig(f"{self.save_direc}totE_fluxes")
        plt.clf()
        plt.close()

        if np.max(abs(self.tot_L_tot-1)) > np.max(abs(self.int_L_tot-1)):
            print(f"Largest del_L_tot = {np.max(abs(self.tot_L_tot-1)):.5f}")
        else:
            print(f"Largest del_L_tot = {np.max(abs(self.int_L_tot-1)):.5f}")

    def PlotRo(self):

        print("### Calculating Ro ###")

        u  = self.snap_tasks['u'][self.SSI:self.SEI,:,:]
        v  = self.snap_tasks['v'][self.SSI:self.SEI,:,:]
        w  = self.snap_tasks['w'][self.SSI:self.SEI,:,:]
        uz = self.snap_tasks['uz'][self.SSI:self.SEI,:,:]
        vz = self.snap_tasks['vz'][self.SSI:self.SEI,:,:]
        wz = self.snap_tasks['wz'][self.SSI:self.SEI,:,:]
        y  = self.snap_scales['y']
        z  = self.snap_scales['z']
        t  = self.snap_t[self.SSI:self.SEI]

        Ly = self.params['Ly']
        lat = self.params['Phi']
        Ta = self.params['Ta']
        Ra = self.params['Ra']

        if Ta == 0:
            print("Ta = 0 therefore cannot compute Ro...")
            return
        print("It didn't quit...")
        del_y = y[1]-y[0]

        inertial_term = np.zeros_like(u)
        coriolis_term = np.zeros_like(u)

        uy = np.zeros_like(u)
        vy = np.zeros_like(u)
        wy = np.zeros_like(u)

        for i in range(len(t)):
            uy[i,:,:] = self.FD_fourth_per(u[i,:,:], del_y)
            vy[i,:,:] = self.FD_fourth_per(v[i,:,:], del_y)
            wy[i,:,:] = self.FD_fourth_per(w[i,:,:], del_y)

        inertial_term = (v*uy + w*uz)**2 + (v*vy + w*vz)**2 + (v*wy + w*wz)**2
        coriolis_term = (w*np.sin(lat) + v*np.cos(lat))**2 + (u*np.sin(lat))**2 + (u*np.cos(lat))**2
        ratio = (inertial_term/(coriolis_term*Ta))**(1/2)

        Ro_t = np.zeros_like(ratio[0,:,:])

        for i in range(len(ratio[0,:,0])):
            for j in range(len(ratio[0,0,:])):
                Ro_t[i,j] = integrate.cumtrapz(ratio[:,i,j], t)[-1]/(t[-1]-t[0])

        self.Ro_z = np.zeros_like(z)
        for i in range(len(self.Ro_z)):
            self.Ro_z[i] = integrate.cumtrapz(Ro_t[:,i], y)[-1]/Ly

        b3i = np.int(np.floor(len(z)/3))
        t3i = np.int(np.floor(2*len(z)/3))

        np.shape(z[b3i:t3i])
        self.Ro_3rd = integrate.cumtrapz(self.Ro_z[b3i:t3i], z[b3i:t3i])[-1]/(z[t3i]-z[b3i])

        plt.plot(self.Ro_z, z)
        plt.xlabel('Ro')
        plt.ylabel('z')
        plt.title(f"$Ro_c = ${(Ra/Ta)**(1/2):.3f}, $Ro_p = ${(Ra/Ta**(3/4))**(1/2):.3f}, $<Ro>_{{[1/3]}} = ${self.Ro_3rd:.3f}")
        plt.savefig(f"{self.save_direc}Ro")
        plt.clf()
        plt.close()

    def E_plots(self):
        '''
        Plot and print values of E.
        '''

        print("### Plotting E ###")

        with h5py.File(f"{self.data_direc}analysis/analysis{self.id}.h5", mode='r') as file:
            self.E_all = np.array(file['tasks']['E_def'])[:,0,0]

        self.mean_E = np.mean(self.E_all[self.ASI:self.AEI], axis=0)

        plt.plot(self.ana_t, self.E_all)
        plt.xlabel(r"t / $\tau_\nu$")
        plt.ylabel("E")
        plt.title(f"<E> = {self.mean_E:.3f}, Time average $= {self.avg_t_range:.3f} \\tau$")
        plt.plot([self.ana_t[self.ASI], self.ana_t[self.ASI]],[0, 1.1*np.max(self.E_all)], 'r', linestyle='--', label='Time average range')
        plt.plot([self.ana_t[self.AEI], self.ana_t[self.AEI]],[0, 1.1*np.max(self.E_all)], 'r', linestyle='--')
        plt.xlim(self.ana_t[0], self.ana_t[-1])
        plt.ylim(0,1.1*np.max(self.E_all))
        plt.legend()

        plt.savefig(f"{self.save_direc}E")
        plt.clf()
        plt.close()

        print(f"E = {self.mean_E:.5f}")

# sim = "r15700"
# folder = "../IVP/EVP-test/n1_t1e5/pert_s/"
# sim_id = sim
#
# direc = f"{folder}/{sim}/raw_data/"
# save  = f"figs/{sim}/"
# data = SimData(direc, save, name=sim, id=sim_id, making_figs=True)
# data.ReadAnalysis()
# data.RuntimeCheck()
# data.AvgIndex(9)
# data.PlotKE()
# data.FluxPlots()
# data.E_plots()
# data.ReadSnapshots()
# # data.PlotRo()s
# data.PlotSnapshots_xyz()
#
# data.SnapshotEvo(cadence=1)
