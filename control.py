class ctrl:

    # control variable containing the key parameters of the simulation. To be passed with most functions
    def __init__(self, N_size, N_grid, N_particles, h, t_step, t_total, t_time, kin_vis):
        self.N_size = N_size                # size of the domain box
        self.N_grid = N_grid                # number of grid point in each direction
        self.N_particles = N_particles      # total number of particles
        self.h = h                          # spatial step size
        self.t_step = t_step                # time step size
        self.t_total = t_total              # total simulation time
        self.t_time = t_time                # current time in the simulation
        self.kin_vis = kin_vis              # kinematic viscosity of the fluid

def count_nonzero(array):
    count=0
    for k in range(0,control.N_grid):
        for j in range(0,control.N_grid):
            for i in range(0,control.N_grid):
                if array[i,j,k,0] != 0:
                    count=count+1
                if array[i,j,k,1] != 0:
                    count=count+1
                if array[i,j,k,2] != 0:
                    count=count+1
    return count

