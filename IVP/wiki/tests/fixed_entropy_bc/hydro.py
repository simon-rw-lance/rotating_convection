"""
Dedalus script for 2D Anelastic Rayleigh-Benard convection.

This script uses a Fourier basis in the x direction with periodic boundary
conditions.

Adapted from the 2D Boussinesq Rayleigh-Benard Convection example
as seen in the Dedalus installation.

Equations are non-dimensionalised by the viscous timescale (t is in units of viscous time).

Can be run using mpi (on for example, 4 processors) by typing

    mpiexec -n 4 python3 anelastic_convection.py

instead of simply

    python3 anelastic_convection.py

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

save_direc = "raw_data/"
pathlib.Path(save_direc).mkdir(parents=True, exist_ok=True)

# Model Parameters
Lx, Lz = 1, 1                       # Domain size
Nx, Nz = 128, 128                   # Resolution
Pr = 1                              # Prandtl number
Ra = 5e4                            # Rayleigh number
Np = 0                              # Number of density scale heights
m = 1.5                             # Polytropic index
theta = 1 - np.exp(-Np/m)           # Dimensionaless inverse T scale height

snapshot_freq = 0.05                # Output frequency of the full data cube
analysis_freq = 0.005               # Output frequency of the analysis tasks
max_dt = 1e-4                       # Maximum allowed timestep

# Initial timestep
init_dt = 1e-5

# Integration parameters --- Note if these are all set to np.inf, simulation will perpetually run.
stop_sim_t  = 1
stop_wall_t = np.inf
stop_iter   = np.inf

# Create bases and domain
x_basis = de.SinCos('x', Nx, interval=(0, Lx), dealias=3/2)   # Fourier basis in the x
z_basis = de.Chebyshev('z', Nz, interval=(0, Lz), dealias=3/2) # Chebyshev basis in the z
domain = de.Domain([x_basis, z_basis], grid_dtype=np.float64)  # Defining our domain
z = domain.grid(1, scales=1)                                   # accessing the z values

# 2D Anelastic hydrodynamics
problem = de.IVP(domain, variables=['p', 's', 'u', 'w', 'sz', 'uz', 'wz', 'L_buoy', 'L_diss'])
problem.meta['p','s','u','w']['z']['dirichlet'] = True
problem.meta['p','s','w','sz','wz','L_buoy','L_diss']['x']['parity'] = 1
problem.meta['u','uz']['x']['parity'] = -1

# Defining model parameters
problem.parameters['Lx'] = Lx
problem.parameters['Lz'] = Lz
problem.parameters['Ra'] = Ra
problem.parameters['Pr'] = Pr
problem.parameters['m'] = m
problem.parameters['theta'] = theta
problem.parameters['X'] = Ra/Pr
problem.parameters['Y'] = (Pr*Pr*theta) / Ra

# Non-constant coeffiecents
rho_ref = domain.new_field(name='rho_ref')
rho_ref['g'] = (1-theta*z)**m
rho_ref.meta['x']['constant'] = True
problem.parameters['rho_ref'] = rho_ref         # Background state for rho
T_ref = domain.new_field(name='T_ref')
T_ref['g'] = 1-theta*z
T_ref.meta['x']['constant'] = True
problem.parameters['T_ref'] = T_ref             # Background state for T
dz_rho_ref = domain.new_field(name='dz_rho_ref')
dz_rho_ref['g'] = -theta*m*((1-theta*z)**(m-1))
dz_rho_ref.meta['x']['constant'] = True
problem.parameters['dz_rho_ref'] = dz_rho_ref   # z-derivative of rho_ref

# Defining d/dz of s, u, and w for reducing our equations to first order
problem.add_equation("sz - dz(s) = 0")
problem.add_equation("uz - dz(u) = 0")
problem.add_equation("wz - dz(w) = 0")

# mass continuity with rho_ref and dz(rho_ref) expanded analytically
problem.add_equation("  (1-theta*z)*(dx(u) + wz) - theta*m*w = 0 ")
# x-component of the momentum equation
problem.add_equation("  rho_ref*( dt(u) - (4/3)*dx(dx(u)) - dz(uz) - (1/3)*dx(wz) ) + dx(p) \
                        - dz_rho_ref*( uz + dx(w) ) \
                        = -rho_ref*( u*dx(u) + w*uz ) ")
# z-component of the momentum equation
problem.add_equation("  rho_ref*T_ref*( dt(w) - X*s - (4/3)*dz(wz) - dx(dx(w)) - (1/3)*dx(uz) ) + T_ref*dz(p) + theta*m*p \
                        + (2/3)*theta*m*rho_ref*( 2*wz - dx(u) ) \
                        = -rho_ref*T_ref*( u*dx(w) + w*wz ) ")
# entropy diffusion equation
problem.add_equation("  T_ref*( Pr*dt(s) - dx(dx(s)) - dz(sz) ) + theta*(m+1)*sz \
                        = -Pr*T_ref*( u*dx(s) + w*sz )    \
                        + 2*Y*( dx(u)*dx(u) + wz*wz + uz*dx(w) - (1/3)*(dx(u)+wz)*(dx(u)+wz) + (1/2)*(uz*uz + dx(w)*dx(w))) ")

# Flux equations for use in analysis outputs
problem.add_equation("  dz(L_buoy) = -s*rho_ref*w")
problem.add_equation("  dz(L_diss) = -2*rho_ref*( (dx(u))**2 + wz**2 + (1/2)*( uz**2 + dx(w)**2 ) + dx(w)*uz - (1/3)*( dx(u) + wz )**2 )")

problem.add_bc("left(w) = 0")            # Impermeable bottom boundary
problem.add_bc("right(w) = 0", condition="(nx != 0)")   # Impermeable top boundary
problem.add_bc("right(p) = 0", condition="(nx == 0)")   # Required for system to be well-posed - see https://bit.ly/2nPVWIg for a related discussion
problem.add_bc("left(uz) = 0")           # Stress-free bottom boundary
problem.add_bc("right(uz) = 0")          # Stress-free top boundary
problem.add_bc("right(s) = 0")           # Fixed entropy at upper boundary, arbitarily set to 0
problem.add_bc("left(sz) = -1")          # Fixed flux at bottom boundary, F = F_cond

problem.add_bc("left(L_buoy) = 0")       # BC for L_buoy for partial depth integration
problem.add_bc("left(L_diss) = 0")       # BC for L_diss for partial depth integration

# Build solver
solver = problem.build_solver(de.timesteppers.RK222)
logger.info('Solver built')

solver.stop_sim_time = stop_sim_t
solver.stop_wall_time = stop_wall_t
solver.stop_iteration = stop_iter

# Initial conditions
z = domain.grid(1)          # accessing z values
s = solver.state['s']       # accessing s values

# Random perturbations, initialized globally for same results in parallel
gshape = domain.dist.grid_layout.global_shape(scales=1)
slices = domain.dist.grid_layout.slices(scales=1)
rand = np.random.RandomState(seed=42)
noise = rand.standard_normal(gshape)[slices]

# Linear background + perturbations damped at walls
zb, zt = z_basis.interval
pert =  1e-5 * noise * (zt - z) * (z - zb)
s['g'] = pert               # assigning our perturbations to s

# CFL criterion
CFL = flow_tools.CFL(solver, initial_dt=init_dt, cadence=10, safety=0.5,
                     max_change=1.5, min_change=0.5, max_dt=max_dt, threshold=0.05)
CFL.add_velocities(('u', 'w'))

# Flow properties
flow = flow_tools.GlobalFlowProperty(solver, cadence=10)
flow.add_property("sqrt(u*u + w*w)", name='Re')

# Saving snapshots
snapshots = solver.evaluator.add_file_handler(f'{save_direc}snapshots', sim_dt=snapshot_freq, max_writes=50)
snapshots.add_system(solver.state)

# Analysis tasks
analysis = solver.evaluator.add_file_handler(f'{save_direc}analysis', sim_dt=analysis_freq, max_writes=5000)
analysis.add_task("integ(s,'x')", layout='g', name='<s>_x')

# Domain averaged Reynolds number
analysis.add_task("integ( integ( sqrt(u*u + w*w) , 'x')/Lx, 'z')/Lz", layout='g', name='Re')

# Flux decomposition - Internal energy equation
analysis.add_task("integ(rho_ref*T_ref*s*w,'x')*Pr/Lx", layout='g', name='L_conv')
analysis.add_task("integ((-1)*rho_ref*T_ref*sz, 'x')/Lx", layout='g', name='L_cond')
analysis.add_task("integ(L_buoy - interp(L_buoy,z=0),'x')*(-Pr*theta)/Lx", layout='g', name='L_buoy')
analysis.add_task("integ(L_diss - interp(L_diss,z=0),'x')*((Pr*Pr*theta)/Ra)/Lx", layout='g', name='L_diss')

# Flux decomposition - Total energy equaton (L_conv and L_cond already outputted)
analysis.add_task("integ(0.5*rho_ref*(u*u + w*w)*w, 'x')*((Pr*Pr*theta)/Ra)/Lx", layout='g', name='L_KE')
analysis.add_task("integ((-1)*rho_ref*(u*(uz + dx(w) ) \
                    + (2/3)*w*(2*wz - dx(u) )), 'x')*((Pr*Pr*theta)/Ra)/Lx", layout='g', name='L_visc')
analysis.add_task("integ(p*w, 'x')*((Pr*Pr*theta)/Ra)/Lx", layout='g', name='L_p')

 # L_enth, the sum of L_conv and L_p
analysis.add_task("(integ(rho_ref*w*T_ref*s, 'x')*Pr + \
                    integ(p*w, 'x')*((Pr*Pr*theta)/Ra))/Lx", layout='g', name='L_enth')

# Magnitude of viscous dissipation as calculated by equation 5 - See Currie & Browning '17
analysis.add_task(" (integ( integ( 2*rho_ref*( \
                        dx(u)**2 + wz**2 \
                        + (1/2)*(uz**2 + dx(w)**2) \
                        + uz*dx(w) \
                        - (1/3)*(dx(u) + wz)**2 \
                        ), 'x'), 'z')/(Lx*Lz) )*((Pr*Pr*theta)/Ra)", layout='g', name='E')

# Domain averaged KE
analysis.add_task(" integ( (integ(0.5*(u*u + w*w)*rho_ref,'x')/Lx), 'z')/Lz", layout='g', name='KE')

# Creating a parameter file
run_parameters = solver.evaluator.add_file_handler(f'{save_direc}run_parameters', wall_dt=1e20, max_writes=1)
run_parameters.add_task(Lx, name="Lx")
run_parameters.add_task(Lz, name="Lz")
run_parameters.add_task(Ra, name="Ra")
run_parameters.add_task(Pr, name="Pr")
run_parameters.add_task(Np, name="Np")
run_parameters.add_task(m, name="m")
run_parameters.add_task(Nx, name="Nx")
run_parameters.add_task(Nz, name="Nz")

run_parameters.add_task(snapshot_freq, name="snap_freq")
run_parameters.add_task(analysis_freq, name="ana_freq")
run_parameters.add_task(max_dt,        name="max_dt")

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
            logger.info('Lx = {}, Lz = {}; (Resolution of {},{})'.format(Lx, Lz, Nx, Nz))
            logger.info('Ra = {}, Pr = {}, Np = {}'.format(Ra, Pr, Np))
            logger.info('Snapshot files outputted every {}'.format(snapshot_freq))
            logger.info('Analysis files outputted every {}'.format(analysis_freq))
            if stop_sim_t != np.inf:
                logger.info('Simulation finishes at sim_time = {}'.format(stop_sim_t))
            elif stop_wall_t != np.inf:
                logger.info('Simulation finishes at wall_time = {}'.format(stop_wall_t))
            elif stop_iter != np.inf:
                logger.info('Simulation finishes at iteration {}'.format(stop_iter))
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
