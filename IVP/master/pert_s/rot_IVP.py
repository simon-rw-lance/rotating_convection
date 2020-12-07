"""
Dedalus script for 2D Anelastic Rayleigh-Benard convection.
This script uses a Fourier basis in the x direction with periodic boundary
conditions.
Adapted from the 2D Boussinesq Rayleigh-Benard Convection example
as seen in the Dedalus installation.
Equations are non-dimensionalised by the viscous timescale (t in units of viscous time).
Can be run using mpi (on for example, 4 processors) by typing
    mpiexec -n 4 python3 rayleigh_benard.py
instead of simply
    python3 rayleigh_benard.py
"""

import numpy as np
from mpi4py import MPI
import time
import matplotlib.pyplot as plt
import sys

from dedalus import public as de
from dedalus.extras import flow_tools
import pathlib

import logging
logger = logging.getLogger(__name__)

import run_param_file as rpf   #Imports a parameter file "run_param_file.py"

save_direc = "raw_data/"
pathlib.Path(save_direc).mkdir(parents=True, exist_ok=True)

# Model Parameters
Ly, Lz = rpf.Lx, rpf.Lz
Ny, Nz = rpf.Nx, rpf.Nz
Pr = rpf.Pr
Ra = rpf.Ra
Np = rpf.Np
Ta = rpf.Ta
Lat = rpf.Phi
m = rpf.m
theta = rpf.theta

# Create bases and domain
y_basis = de.Fourier('y', Ny, interval=(0, Ly), dealias=3/2)   # Fourier basis in the x
z_basis = de.Chebyshev('z', Nz, interval=(0, Lz), dealias=3/2) # Chebyshev basis in the z
domain = de.Domain([y_basis, z_basis], grid_dtype=np.float64)  # Defining our domain
z = domain.grid(1, scales=1)                                   # accessing the z values

# 2D Anelastic hydrodynamics
problem = de.IVP(domain, variables=['p', 's', 'u', 'v', 'w', 'sz', 'uz', 'vz', 'wz', 'L_buoy', 'L_diss'])
problem.meta['p','s','u','w']['z']['dirichlet'] = True

# Defining model parameters
problem.parameters['Ly'] = Ly
problem.parameters['Lz'] = Lz
problem.parameters['Ra'] = Ra
problem.parameters['Pr'] = Pr
problem.parameters['Ta'] = Ta
problem.parameters['Lat'] = Lat
problem.parameters['m'] = m
problem.parameters['theta'] = theta
problem.parameters['X'] = Ra/Pr
problem.parameters['Y'] = (Pr*Pr*theta) / Ra
problem.parameters['T'] = Ta**(1/2)
problem.substitutions['s_total'] = 's + ( 1/(theta*m*rho_ref) ) * ( rho_ref*(1-theta)**(-m) - 1 )'
problem.substitutions['s_total_z'] = 'sz - 1/(rho_ref*T_ref)'

# Non-constant coeffiecents
rho_ref = domain.new_field(name='rho_ref')
rho_ref['g'] = (1-theta*z)**m
rho_ref.meta['y']['constant'] = True
problem.parameters['rho_ref'] = rho_ref         # Background state for rho
T_ref = domain.new_field(name='T_ref')
T_ref['g'] = 1-theta*z
T_ref.meta['y']['constant'] = True
problem.parameters['T_ref'] = T_ref             # Background state for T
dz_rho_ref = domain.new_field(name='dz_rho_ref')
dz_rho_ref['g'] = -theta*m*((1-theta*z)**(m-1))
dz_rho_ref.meta['y']['constant'] = True
problem.parameters['dz_rho_ref'] = dz_rho_ref   # z-derivative of rho_ref

# Defining d/dz of s, u, and w for reducing our equations to first order
problem.add_equation("sz - dz(s) = 0")
problem.add_equation("uz - dz(u) = 0")
problem.add_equation("vz - dz(v) = 0")
problem.add_equation("wz - dz(w) = 0")

# mass continuity with rho_ref and dz(rho_ref) expanded analytically
problem.add_equation("  T_ref*(dy(v) + wz) - theta*m*w = 0 ")

# x-component of the momentum equation
problem.add_equation("  rho_ref*( dt(u) - dy(dy(u)) - dz(uz) + T*(w*cos(Lat) - v*sin(Lat)) ) - dz_rho_ref*uz \
                        = -rho_ref*( v*dy(u) + w*uz ) ")

# y-component of the momentum equation
problem.add_equation("  rho_ref*( dt(v) - (4/3)*dy(dy(v)) - dz(vz) - (1/3)*dy(wz) + T*u*sin(Lat) ) + dy(p) - dz_rho_ref*(vz + dy(w)) \
                        = -rho_ref*( v*dy(v) + w*vz )")

# z-component of the momentum equation
problem.add_equation("  rho_ref*T_ref*( dt(w) - X*s - dy(dy(w)) - (4/3)*dz(wz) - (1/3)*dy(vz) - T*u*cos(Lat) ) \
                        + T_ref*dz(p) + theta*m*p + (2/3)*theta*m*rho_ref*( 2*wz - dy(v) ) \
                        = -rho_ref*T_ref*( v*dy(w) + w*wz )")

# entropy diffusion equation
problem.add_equation("  T_ref*( Pr*dt(s) - dy(dy(s)) - dz(sz) ) - Pr*w/rho_ref + theta*(m+1)*sz \
                        = -Pr*T_ref*( v*dy(s) + w*sz ) \
                        + 2*Y*( dy(v)*dy(v) + wz*wz + vz*dy(w) - (1/3)*(dy(v) + wz)*(dy(v) + wz) + (1/2)*(dy(u)*dy(u) + uz*uz + vz*vz + dy(w)*dy(w)) )")

# Flux equations for use in analysis outputs
problem.add_equation("  dz(L_buoy) = -s_total*rho_ref*w")
problem.add_equation("  dz(L_diss) = -2*rho_ref*( dy(v)*dy(v) + wz*wz + vz*dy(w) - (1/3)*(dy(v) + wz)*(dy(v) + wz) + (1/2)*(dy(u)*dy(u) + uz*uz + vz*vz + dy(w)*dy(w)) )")

problem.add_bc("left(w) = 0")            # Impermeable bottom boundary
problem.add_bc("right(w) = 0", condition="(ny != 0)")   # Impermeable top boundary
problem.add_bc("right(p) = 0", condition="(ny == 0)")   # Required for equations to be well-posed - see https://bit.ly/2nPVWIg for a related discussion
problem.add_bc("left(uz) = 0")           # Stress-free bottom boundary
problem.add_bc("right(uz) = 0")          # Stress-free top boundary
problem.add_bc("left(vz) = 0")
problem.add_bc("right(vz) = 0")
problem.add_bc("right(s) = 0")           # Fixed entropy at upper boundary, arbitarily set to 0
problem.add_bc("left(sz) = 0")          #Fixed flux at bottom boundary, F = F_cond

problem.add_bc("left(L_buoy) = 0")       # BC for L_buoy for partial depth integration
problem.add_bc("left(L_diss) = 0")       # BC for L_diss for partial depth integration

# Build solver
solver = problem.build_solver(de.timesteppers.RK222)
logger.info('Solver built')

# write, dt = solver.load_state(data_direc, -1)

# Initial conditions
x = domain.grid(0)
z = domain.grid(1)
s = solver.state['s']
w = solver.state['w']
sz = solver.state['sz']

# Random perturbations, initialized globally for same results in parallel
gshape = domain.dist.grid_layout.global_shape(scales=1)
slices = domain.dist.grid_layout.slices(scales=1)
rand = np.random.RandomState(seed=42)
noise = rand.standard_normal(gshape)[slices]

# Linear background + perturbations damped at walls
zb, zt = z_basis.interval
pert =  1e-5 * noise * (zt - z) * (z - zb)
s['g'] = pert
s.differentiate('z', out=sz)

# Initial timestep
dt = rpf.initial_timestep

# Integration parameters --- Note if these are all set to np.inf, simulation will perpetually run.
solver.stop_sim_time = rpf.end_sim_time
solver.stop_wall_time = rpf.end_wall_time
solver.stop_iteration = rpf.end_iterations

# CFL criterion
CFL = flow_tools.CFL(solver, initial_dt=dt, cadence=10, safety=0.5,
                     max_change=1.5, min_change=0.5, max_dt=rpf.max_dt, threshold=0.05)
CFL.add_velocities(('v', 'w'))

# Flow properties
flow = flow_tools.GlobalFlowProperty(solver, cadence=10)
flow.add_property("sqrt(u*u + v*v + w*w)", name='Re')

# Saving snapshots
snapshots = solver.evaluator.add_file_handler(save_direc + 'snapshots', sim_dt=rpf.snapshot_freq, max_writes=50)
snapshots.add_system(solver.state)

# Analysis tasks
analysis = solver.evaluator.add_file_handler(save_direc + 'analysis', sim_dt=rpf.analysis_freq, max_writes=5000)
analysis.add_task("integ(s_total,'y')/Ly", layout='g', name='<s>_y')

# Mean Reynolds number
analysis.add_task("integ( integ( sqrt(u*u + v*v + w*w) , 'y')/Ly, 'z')/Lz", layout='g', name='Re')

# Reynolds stresses (averaged over y):
analysis.add_task("integ( ((u - (integ(u, 'y')/Ly)) * (v - (integ(v, 'y')/Ly))), 'y')/Ly", layout='g', name='RS_xy')
analysis.add_task("integ( ((u - (integ(u, 'y')/Ly)) * (w - (integ(w, 'y')/Ly))), 'y')/Ly", layout='g', name='RS_xz')
analysis.add_task("integ( ((v - (integ(v, 'y')/Ly)) * (w - (integ(w, 'y')/Ly))), 'y')/Ly", layout='g', name='RS_yz')

# Differential Reynolds stresses:
analysis.add_task("dz( integ( ((u - (integ(u, 'y')/Ly)) * (v - (integ(v, 'y')/Ly))), 'y')/Ly )", layout='g', name='RS_xy_dz')
analysis.add_task("dz( integ( ((u - (integ(u, 'y')/Ly)) * (w - (integ(w, 'y')/Ly))), 'y')/Ly )", layout='g', name='RS_xz_dz')
analysis.add_task("dz( integ( ((v - (integ(v, 'y')/Ly)) * (w - (integ(w, 'y')/Ly))), 'y')/Ly )", layout='g', name='RS_yz_dz')

# Pressure and pressure differential:
analysis.add_task(" integ(rho_ref, 'y') / Ly", layout='g', name='p')
analysis.add_task(" integ(dz_rho_ref, 'y') / Ly", layout='g', name='dp')

# Mean flows:
analysis.add_task("integ( u, 'y' )/Ly", layout='g', name='u_bar')
analysis.add_task("integ( v, 'y' )/Ly", layout='g', name='v_bar')
analysis.add_task("integ( w, 'y' )/Ly", layout='g', name='w_bar')

# Rossby number
if Ta == 0:
    analysis.add_task( np.inf, layout='g', name='Ro_layer' )
else:
    analysis.add_task(" integ( sqrt( (dy(w) - vz)**2 + uz**2 + dy(u)**2 ) / T, 'y' ) / Ly", layout='g', name='Ro_layer')

#Flux decomposition - Internal energy equation
analysis.add_task("integ(rho_ref*T_ref*s_total*w,'y')*Pr/Ly", layout='g', name='L_conv')
analysis.add_task("integ((-1)*rho_ref*T_ref*s_total_z, 'y')/Ly", layout='g', name='L_cond')
analysis.add_task("integ(L_buoy - interp(L_buoy,z=0),'y')*(-Pr*theta)/Ly", layout='g', name='L_buoy')
analysis.add_task("integ(L_diss - interp(L_diss,z=0),'y')*((Pr*Pr*theta)/Ra)/Ly", layout='g', name='L_diss')

#Flux decomposition - Total energy equaton (L_conv and L_cond already outputted)
analysis.add_task("integ(0.5*rho_ref*(u*u + v*v + w*w)*w, 'y')*((Pr*Pr*theta)/Ra)/Ly", layout='g', name='L_KE')
analysis.add_task("integ( (-1)*rho_ref*( u*uz + v*( vz+dy(w) ) + (2/3)*w*( 2*wz - dy(v) ) ), 'y')*((Pr*Pr*theta)/Ra)/Ly",layout='g',name='L_visc')
analysis.add_task("integ(p*w, 'y')*((Pr*Pr*theta)/Ra)/Ly", layout='g', name='L_p')

#L_enth, the sum of L_conv and L_p
analysis.add_task("(integ(rho_ref*w*T_ref*s_total, 'y')*Pr + \
                   integ(p*w, 'y')*((Pr*Pr*theta)/Ra))/Ly", layout='g', name='L_enth')

#Magnitude of viscous dissipation as calculated by equation 5 (E_def) and equation 24 (E_F_conv) - See C&B '17
analysis.add_task(" integ( integ( 2*rho_ref*( dy(v)*dy(v) + wz*wz + vz*dy(w) - (1/3)*(dy(v)+wz)*(dy(v)+wz) + (1/2)*(dy(u)*dy(u) + uz*uz + vz*vz + dy(w)*dy(w)) ) \
                               , 'y'), 'z')*(((Pr*Pr*theta)/Ra )/(Ly*Lz)) ", layout='g', name='E_def')
analysis.add_task(" integ( (integ(rho_ref*T_ref*s*w,'y')/Ly)/T_ref, 'z')*Pr*theta/Lz ", layout='g', name='E_F_conv')

#E_def as a function of the z direction
analysis.add_task(" integ( 2*rho_ref*( dy(v)*dy(v) + wz*wz + vz*dy(w) - (1/3)*(dy(v)+wz)*(dy(v)+wz) + (1/2)*(dy(u)*dy(u) + uz*uz + vz*vz + dy(w)*dy(w)) ) \
                                       , 'y')*(((Pr*Pr*theta)/Ra )/(Ly)) ", layout='g', name='E_def_z')

# Mean KE
analysis.add_task(" integ( (integ(0.5*(u*u + v*v + w*w)*rho_ref,'y')/Ly), 'z')/Lz", layout='g', name='KE')

# Vertical momentum profiles
analysis.add_task("integ( rho_ref*u, 'y')/Ly", layout='g', name='<x_mom>_y')
analysis.add_task("integ( rho_ref*v, 'y')/Ly", layout='g', name='<y_mom>_y')
analysis.add_task("integ( rho_ref*w, 'y')/Ly", layout='g', name='<z_mom>_y')

# x-momentum
analysis.add_task(" integ( integ( rho_ref*u, 'y')/Ly, 'z')/Lz", layout='g', name='<x_mom>')
analysis.add_task(" integ( integ( rho_ref*( dy(dy(u)) + dz(uz) ), 'y')/Ly, 'z')/Lz",         layout='g', name='<x_mom>_visc1')
analysis.add_task(" integ( integ( -rho_ref*T*( w*cos(Lat) + v*sin(Lat) ), 'y')/Ly, 'z')/Lz", layout='g', name='<x_mom>_rot')
analysis.add_task(" integ( integ( dz_rho_ref*uz, 'y')/Ly, 'z')/Lz",                          layout='g', name='<x_mom>_visc2')
analysis.add_task(" integ( integ( -rho_ref*( v*dy(u) + w*uz ), 'y')/Ly, 'z')/Lz",            layout='g', name='<x_mom>_advec')

# y-momentum
analysis.add_task(" integ( integ( rho_ref*v, 'y')/Ly, 'z')/Lz", layout='g', name='<y_mom>')
analysis.add_task(" integ( integ( rho_ref*( (4/3)*dy(dy(v)) + dz(vz) + (1/3)*dy(vz) ), 'y')/Ly, 'z')/Lz", layout='g', name='<y_mom>_visc1')
analysis.add_task(" integ( integ( -rho_ref*T*u*sin(Lat), 'y')/Ly, 'z')/Lz",                               layout='g', name='<y_mom>_rot')
analysis.add_task(" integ( integ( -dy(p), 'y')/Ly, 'z')/Lz",                                              layout='g', name='<y_mom>_gradP')
analysis.add_task(" integ( integ( dz_rho_ref*( vz + dy(w) ), 'y')/Ly, 'z')/Lz",                           layout='g', name='<y_mom>_visc2')
analysis.add_task(" integ( integ( rho_ref*( v*dy(v) + w*vz ), 'y')/Ly, 'z')/Lz",                          layout='g', name='<y_mom>_advec')

# z-momentum
analysis.add_task(" integ( integ( rho_ref*w, 'y')/Ly, 'z')/Lz", layout='g', name='<z_mom>')
analysis.add_task(" integ( integ( rho_ref*X*s, 'y')/Ly, 'z')/Lz",                                         layout='g', name='<z_mom>_buoy')
analysis.add_task(" integ( integ( rho_ref*( dy(dy(w)) + (4/3)*dz(wz) + (1/3)*dy(vz) ), 'y')/Ly, 'z')/Lz", layout='g', name='<z_mom>_visc1')
analysis.add_task(" integ( integ( rho_ref*T*u*cos(Lat), 'y')/Ly, 'z')/Lz",                                layout='g', name='<z_mom>_rot')
analysis.add_task(" integ( integ( -dz(p), 'y')/Ly, 'z')/Lz",                                              layout='g', name='<z_mom>_gradP')
analysis.add_task(" integ( integ( -theta*m*p/T_ref, 'y')/Ly, 'z')/Lz",                                    layout='g', name='<z_mom>_P2')
analysis.add_task(" integ( integ( -(2/3)*theta*m*rho_ref/T_ref, 'y')/Ly, 'z')/Lz",                        layout='g', name='<z_mom>_visc2')
analysis.add_task(" integ( integ( -rho_ref*( v*dy(w) + w*wz ), 'y')/Ly, 'z')/Lz",                         layout='g', name='<z_mom>_advec')

# Creating a parameter file
run_parameters = solver.evaluator.add_file_handler(save_direc + 'run_parameters', wall_dt=1e20, max_writes=1)
run_parameters.add_task(Ly, name="Ly")
run_parameters.add_task(Lz, name="Lz")
run_parameters.add_task(Ra, name="Ra")
run_parameters.add_task(Pr, name="Pr")
run_parameters.add_task(Np, name="Np")
run_parameters.add_task(m,  name="m")
run_parameters.add_task(Ny, name="Ny")
run_parameters.add_task(Nz, name="Nz")
run_parameters.add_task(Ta, name="Ta")
run_parameters.add_task(Lat, name="Phi")

run_parameters.add_task(rpf.snapshot_freq, name="snap_freq")
run_parameters.add_task(rpf.analysis_freq, name="ana_freq")
run_parameters.add_task(rpf.max_dt,        name="max_dt")

# Main loop
try:
    logger.info('Starting loop')
    start_time = time.time()
    while solver.ok:
        dt = CFL.compute_dt()
        dt = solver.step(dt)

        if (solver.iteration) == 1:
            # Prints various parameters to terminal upon starting the simulation
            logger.info('Parameter values imported form run_param_file.py:')
            logger.info('Ly = {}, Lz = {}; (Resolution of {},{})'.format(Ly, Lz, Ny, Nz))
            logger.info('Ra = {}, Pr = {}, Np = {}'.format(Ra, Pr, Np))
            logger.info('Ta = {}, Phi = {}'.format(Ta, Lat))
            logger.info('Snapshot files outputted every {}'.format(rpf.snapshot_freq))
            logger.info('Analysis files outputted every {}'.format(rpf.analysis_freq))
            if rpf.end_sim_time != np.inf:
                logger.info('Simulation finishes at sim_time = {}'.format(rpf.end_sim_time))
            elif rpf.end_wall_time != np.inf:
                logger.info('Simulation finishes at wall_time = {}'.format(rpf.end_wall_time))
            elif rpf.end_iterations != np.inf:
                logger.info('Simulation finishes at iteration {}'.format(rpf.end_iterations))
            else:
                logger.info('No clear end point defined. Simulation may run perpetually.')

        if (solver.iteration-1) % 10 == 0:
            # Prints progress information include maximum Reynolds number every 10 iterations
            logger.info('Iteration: %i, Time: %e, dt: %e' %(solver.iteration, solver.sim_time, dt))
            logger.info('Max Re = %f' %flow.max('Re'))

except:
    logger.error('Exception raised, triggering end of main loop.')
    raise
finally:
    # Prints concluding information upon reaching the end of the simulation.
    end_time = time.time()
    logger.info('Iterations: %i' %solver.iteration)
    logger.info('Sim end time: %f' %solver.sim_time)
    logger.info('Run time: %.2f sec' %(end_time-start_time))
    logger.info('Run time: %f cpu-hr' %((end_time-start_time)/60/60*domain.dist.comm_cart.size))
