"""
Parameter file for use in the Dedalus 2D anelastic convection script.
"""

import numpy as np

Lx, Lz = 2, 1                       # Domain size
Nx, Nz = 128, 64                    # Number of
Pr = 1.                             # Prandtl number
Ra = 15700                          # Rayleigh number
Ta = 1e5
Np = 1                              # Number of density scale heights
m = 1.5                             # Polytropic index
Phi = np.pi/4
theta = 1 - np.exp(-Np/m)           # Dimensionaless inverse T scale height

initial_timestep = 1.5e-5           # Initial timestep
max_dt = 5e-3                       # max dt

snapshot_freq = 1.5e-1              # Frequency snapshot files are outputted
analysis_freq = 5e-3              # Frequency analysis files are outputted

end_sim_time = 10.                   # Stop time in simulations units
end_wall_time = np.inf              # Stop time in wall time
end_iterations = np.inf             # Stop time in iterations
