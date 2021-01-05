
"""
Finds the critical Rayleigh number and wavenumber for the 2-dimensional,
incompressible, Boussinesq Navier-Stokes equations in order to determine
the onset of convection in such a system.
"""
import matplotlib.pyplot as plt
# matplotlib.use('Agg')
from mpi4py import MPI
from eigentools import Eigenproblem, CriticalFinder
import time
import dedalus.public as de
import numpy as np
import cmath
import pathlib
from scipy import interpolate, optimize

comm = MPI.COMM_WORLD

Nz = 64
Np = 0.5
Ta = 1e4
m = 1.5
theta = theta = 1 - np.exp(-Np/m)

# Creating parameter space
Ra_vals, ky_vals = 45, 45
mins = np.array((4000, 1))
maxs = np.array((15000, 12))
nums = np.array((Ra_vals, ky_vals))

# Restricted horizontal extent parameters
Lx=2
n=1
kx_restrict = []

# folder = 'check_Ra_crit/rotational/n0-5/t1e5/'
# file_name = f'gr_np_{Np}_Ta_{int(Ta)}'
# direc = f'{folder}{file_name}'

direc = f'gr_np_{Np}_Ta_{int(Ta)}'
direc = direc.replace('.', '-')

z_basis = de.Chebyshev('z',Nz, interval=(0, 1))
d = de.Domain([z_basis],comm=MPI.COMM_SELF)
z = z_basis.grid()

rayleigh_benard = de.EVP(d,['p', 's', 'u', 'v', 'w', 'sz', 'uz', 'vz', 'wz'], eigenvalue='omega')
# Defining model parameters
rayleigh_benard.parameters['Ra'] = Ra = 1000
rayleigh_benard.parameters['Pr'] = Pr = 1
rayleigh_benard.parameters['T'] = Ta**(1/2)
rayleigh_benard.parameters['Lat'] = np.pi/4
rayleigh_benard.parameters['m'] = m
rayleigh_benard.parameters['theta'] = theta
rayleigh_benard.substitutions['X'] = 'Ra/Pr'
rayleigh_benard.substitutions['Y'] = '(Pr*Pr*theta) / Ra'
rayleigh_benard.parameters['k'] = 1
rayleigh_benard.substitutions['dt(A)'] = 'omega*A'
rayleigh_benard.substitutions['dy(A)'] = '1j*k*A'

# Non-constant coeffiecents
rho_ref = d.new_field(name='rho_ref')
rho_ref['g'] = (1-theta*z)**m
rayleigh_benard.parameters['rho_ref'] = rho_ref         # Background state for rho

T_ref = d.new_field(name='T_ref')
T_ref['g'] = 1-theta*z
rayleigh_benard.parameters['T_ref'] = T_ref             # Background state for T

dz_rho_ref = d.new_field(name='dz_rho_ref')
dz_rho_ref['g'] = -theta*m*((1-theta*z)**(m-1))
rayleigh_benard.parameters['dz_rho_ref'] = dz_rho_ref   # z-derivative of rho_ref

dz_sb = d.new_field(name='dz_sb')
dz_sb['g'] =  -1/(T_ref['g']*rho_ref['g'])
rayleigh_benard.parameters['dz_sb'] = dz_sb

# Defining d/dz of s, u, and w for reducing our equations to first order
rayleigh_benard.add_equation("sz - dz(s) = 0")
rayleigh_benard.add_equation("uz - dz(u) = 0")
rayleigh_benard.add_equation("vz - dz(v) = 0")
rayleigh_benard.add_equation("wz - dz(w) = 0")

# mass continuity with rho_ref and dz(rho_ref) expanded analytically
rayleigh_benard.add_equation("  (1-theta*z)*(dy(v) + wz) - theta*m*w = 0 ")

# x-component of the momentum equation
rayleigh_benard.add_equation("  rho_ref*( dt(u) - dy(dy(u)) - dz(uz) + T*(w*cos(Lat) - v*sin(Lat)) ) - dz_rho_ref*uz \
                        = -rho_ref*( v*dy(u) + w*uz ) ")

# y-component of the momentum equation
rayleigh_benard.add_equation("  rho_ref*( dt(v) - (4/3)*dy(dy(v)) - dz(vz) - (1/3)*dy(wz) + T*u*sin(Lat) ) + dy(p) - dz_rho_ref*(vz + dy(w)) \
                        = -rho_ref*( v*dy(v) + w*vz )")

# z-component of the momentum equation
rayleigh_benard.add_equation("  rho_ref*T_ref*( dt(w) - X*s - dy(dy(w)) - (4/3)*dz(wz) - (1/3)*dy(vz) - T*u*cos(Lat) ) \
                        + T_ref*dz(p) + theta*m*p + (2/3)*theta*m*rho_ref*( 2*wz - dy(v) ) \
                        = -rho_ref*T_ref*( v*dy(w) + w*wz )")

# entropy diffusion equation
rayleigh_benard.add_equation("  T_ref*( Pr*dt(s) - dy(dy(s)) - dz(sz) ) - Pr*w/rho_ref + theta*(m+1)*sz \
                        = -Pr*T_ref*( v*dy(s) + w*sz ) \
                        + 2*Y*( dy(v)*dy(v) + wz*wz + vz*dy(w) - (1/3)*(dy(v) + wz)*(dy(v) + wz) + (1/2)*(dy(u)*dy(u) + uz*uz + vz*vz + dy(w)*dy(w)) )")


rayleigh_benard.add_bc("right(s) = 0")           # Fixed entropy at upper boundary, arbitarily set to 0
rayleigh_benard.add_bc("left(sz) = 0")          # Fixed flux at bottom boundary, F = F_cond

#Impenetrable
rayleigh_benard.add_bc('left(w) = 0')
rayleigh_benard.add_bc('right(w) = 0')

#Stress free
rayleigh_benard.add_bc('left(uz) = 0')
rayleigh_benard.add_bc('right(uz) = 0')
rayleigh_benard.add_bc("left(vz) = 0")
rayleigh_benard.add_bc("right(vz) = 0")

# create an Eigenproblem object
EP = Eigenproblem(rayleigh_benard, sparse=True)

# create a shim function to translate (x, y) to the parameters for the eigenvalue problem:
def shim(x,y):
    # print(f"Processor: {comm.rank} \n Ra={int(x)}, k={y:.3}")
    gr, indx, freq = EP.growth_rate({"Ra":x,"k":y})
    # print(f"Processor: {comm.rank} \n Ra={int(x)}, k={y:.3}, gr={gr}")
    ret = gr+1j*freq
    if type(ret) == np.ndarray:
        return ret[0]
    else:
        return ret

cf = CriticalFinder(shim, comm)

if comm.rank == 0:
    print('\n### Generating Grid ###\n')
start = time.time()

# Load in data file if previously run (commented out while fiddling)
try:
    cf.load_grid(f'{direc}.h5')
except:
    # Generates the grid of growth rates (solves the EVP's here)
    # generating the grid is the longest part
    cf.grid_generator(mins, maxs, nums)
    if comm.rank == 0:
        cf.save_grid(direc)

# cf.grid_generator(mins, maxs, nums)
# if comm.rank == 0:
#     print('\nSaving grid...')
#     cf.save_grid(direc)

end = time.time()
if comm.rank == 0:
    print('### Grid Generation Complete ###')
    print(f"Time taken: {(end-start)/60:10.5f} m ({end-start:10.5f} s) \n")

# Searches the grid of growth rates for each of kx finding the Ra where growth
# growth rates pass through zero. Returns NaN for a given kx if no root found.
cf.root_finder()

# r_grid = cf.xyz_grids[0]
# k_grid = cf.xyz_grids[1]
# np.max(cf.grid.real.T)
# # print(cf.grid.real.T)
#
#
# dead_vals = np.where(cf.grid.real.T >= 1e20)
#
# new_grid = np.copy(cf.grid.real.T)
# print(np.max(new_grid))
# print(np.max(cf.grid.real.T))
#
# for i in range(len(dead_vals)):
#     new_grid[dead_vals[0][i],dead_vals[1][i]] = np.nan
# print(np.max(new_grid))
# print(np.max(cf.grid.real.T))
#
# biggest_val = np.nanstd(2*np.abs(new_grid))
# cmap = plt.cm.get_cmap('RdBu_r')
# cmap.set_bad('magenta')
# plt.pcolormesh(k_grid,r_grid,new_grid.T,cmap=cmap,vmin=-biggest_val,vmax=biggest_val)
# # plt.pcolormesh(k_grid,r_grid,cf.grid.real,cmap='RdBu_r',vmin=-biggest_val,vmax=biggest_val)
# plt.colorbar()
# plt.show()

# np.max(new_grid)
# Interpolates the non-NaN roots and minimises the function to find Ra_crit
# (Note this allows for all values of kx, e.g. unrestricted horizontal domain)
crit = cf.crit_finder(find_freq = True)

if comm.rank == 0:
    # Print overall results
    print("crit = {}".format(crit))
    print("critical wavenumber k = {:10.5f}".format(crit[1]))
    print("critical Ra = {:10.5f}".format(crit[0]))
    print("critical freq = {:10.5f}".format(crit[2]))

    # Find specific Ra_c for a given horizontal extend (Lx)
    mask = np.isfinite(cf.roots)
    good_values = [array[0,mask] for array in cf.xyz_grids[1:]]
    rroot = cf.roots[mask] # Masking NaN values
    root_fn = interpolate.interp1d(good_values[0],rroot,kind='cubic') # Interpolating over masked roots
    x_fn = np.linspace(good_values[0][0], good_values[0][-1], 100) # Constructing a kx array for plotting

    # Restrict values of kx to those which fit in our box shape Lx
    while (n*2*np.pi/Lx < good_values[0][-1]):
        if (n*2*np.pi/Lx < good_values[0][0]):
            pass
        else:
            kx_restrict.append(n*2*np.pi/Lx)
        n += 1
    print(f"Restricted kx values for Lx = {Lx}:")
    print(kx_restrict)
    Ra_restrict = root_fn(kx_restrict)

    index = np.where(Ra_restrict == np.min(Ra_restrict))[0][0]

    fig = plt.figure(figsize=(9,6))
    ax = fig.add_subplot(111)
    xx = cf.xyz_grids[1].T
    yy = cf.xyz_grids[0].T
    grid = cf.grid.real.T
    biggest_val = 2*np.abs(grid).std()
    plt.pcolormesh(xx,yy,grid,cmap='RdBu_r',vmin=-biggest_val,vmax=biggest_val)
    plt.colorbar()
    x = cf.xyz_grids[1][0,:]
    y = cf.roots[:]
    plt.scatter(x,y)
    plt.plot(x_fn,root_fn(x_fn), ':')
    plt.ylim(yy.min(),yy.max())
    plt.xlim(xx.min(),xx.max())
    plt.xlabel("kx")
    plt.ylabel("Ra")

    plt.title("All kx: Ra_c = {:.2f}, kx_c = {:.3f} \n Restricted kx: Lx = {}, Ra_c = {:.2f}, kx_c = {:.3f} \n ".format(crit[0],crit[1], Lx, Ra_restrict[index], kx_restrict[index]))
    plt.tight_layout()
    plt.savefig(direc + ".png")
