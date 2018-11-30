import numpy as np
import scipy as sp
import scipy.interpolate
import matplotlib as mpl
import matplotlib.pylab as plt
from mpl_toolkits.mplot3d import axes3d
from sys import getsizeof
import datetime
import time
from timeit import default_timer as timer
import pycuda as cuda
import pycuda.driver as drv
import pycuda.gpuarray as gpuarray
import pycuda.autoinit
from pycuda.compiler import SourceModule
import os
import sys

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


class ring_val:

    # control variable containing the parameters of one vortex ring.
    def __init__(self, R, r, x_pos, y_pos, z_pos, Re, N_th, N_phi, N_d):
        self.R = R              # main ring radius
        self.r = r              # ring tube radius
        self.x_pos = x_pos      # x position of the center of the ring
        self.y_pos = y_pos      # y position of the center of the ring
        self.z_pos = z_pos      # z position of the center of the ring
        self.Re = Re            # Reynolds number of the vortex ring
        self.N_th = N_th        # number of ring tube slices
        self.N_phi = N_phi      # number of main ring slices
        self.N_d = N_d          # number radial points


#################################   FUNCTION-DECLARATION   #################################


def phi(x):
    
    # interpolation function M4'-Spline
    if abs(x) > 2:
        return 0
    elif abs(x) <= 1:
        phi_x = (2 - 5 * pow(x,2) + 3 * pow(abs(x),3)) / 2
    elif abs(x) > 1 and abs(x) <= 2:
        phi_x = pow(2 - abs(x),2) * (1 - abs(x)) / 2

    return phi_x


def c_ring(R, r, w_0, Re, alpha, control):

    # create counting variables and poistion of the ring
    count_ring = 0
    N = 0
    x_pos = np.pi
    y_pos = np.pi
    z_pos = np.pi
    Re = 1000

    # loop through all the grid nodes
    for k in range(0,control.N_grid):
        for j in range(0,control.N_grid):
            for i in range(0,control.N_grid):

                # assign position to the particles
                alpha[N,0,0] = i * control.h
                alpha[N,1,0] = j * control.h
                alpha[N,2,0] = k * control.h

                # condition for particle to gain initial vorticity
                if (((i * control.h - x_pos) ** 2 + (j * control.h - y_pos) ** 2) ** 0.5 - R) ** 2 + (k * control.h - z_pos) ** 2 < r ** 2:

                    # calculate some values needed for vorticity
                    gam = Re * control.kin_vis
                    rad = ((i * control.h) ** 2 + (j * control.h) ** 2) ** 0.5
                    rho = ((rad - R) ** 2 + (k * control.h) ** 2) ** 0.5
                    th = np.arctan2((j * control.h), (i * control.h))
                    
                    # calculate magnitude of vorticity
                    w_mag = gam / (np.pi * R ** 2) * np.exp(-(rho / r) ** 2)

                    # vectorize vorticity
                    alpha[N,0,1] = np.sin(th) * w_mag
                    alpha[N,1,1] = -np.cos(th) * w_mag
                    count_ring = count_ring + 1
                N = N + 1
    
    # print out counting variable and plot the ring
    print count_ring, N
    """fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(alpha[:,0,0], alpha[:,1,0], alpha[:,2,0], c='r', marker='.')
    ax.set_xlim3d(0,control.N_size)
    ax.set_ylim3d(0,control.N_size)
    ax.set_zlim3d(0,control.N_size)
    plt.show()"""       #### probably not needed ###


def c_ring2(ring, alpha, control):  # create the particles for one vortex ring given the parameters passed with "ring"

    # create counting and loop variables
    N = 0
    th = np.linspace(0,2 * np.pi,ring.N_th + 1)
    phi = np.linspace(0,2 * np.pi,ring.N_phi + 1)
    d = np.linspace(ring.r / 10,ring.r,ring.N_d)

    # loop through all positions in the ring
    for k in range(0,ring.N_d):
        for j in range(0,ring.N_phi):
            for i in range(0,ring.N_th):

                # assign positions to the particles
                alpha[N,0,0] = ring.x_pos + (ring.R + d[k] * np.cos(th[i])) * np.cos(phi[j])
                alpha[N,1,0] = ring.y_pos + (ring.R + d[k] * np.cos(th[i])) * np.sin(phi[j])
                alpha[N,2,0] = ring.z_pos + d[k] * np.sin(th[i])

                # calculate the mangitude of vorticity
                gam = ring.Re * control.kin_vis
                w_mag = gam / (np.pi * ring.R ** 2) * np.exp(-(d[k] / ring.r) ** 2)

                # vectorize vorticity
                alpha[N,0,3] = np.sin(th[i]) * w_mag
                alpha[N,1,3] = -np.cos(th[i]) * w_mag
                N = N + 1                
    
    # print out counting variable
    print "%d particles created" %N
      
                  
def inter_P2G(alpha,w,control): # interpolate the strengths of the particles onto the grid

    # start the timer for the whole function
    start1 = timer()

    # loop through all the grid nodes
    for k in range(0,control.N_grid):
        for j in range(0,control.N_grid):

            # start the timer for the current yz-grid step
            start = timer()
            for i in range(0,control.N_grid):

                # set the summation variable to 0
                w_sum0 = 0
                w_sum1 = 0
                w_sum2 = 0

                # loop through all particle positions
                for N in range(0,control.N_particles):

                    # check if one of the interpolations is zero. If yes jump to the next particle.
                    if phi((i * control.h - alpha[N,0,0]) / control.h) == 0:
                        continue
                    if phi((j * control.h - alpha[N,1,0]) / control.h) == 0:
                        continue
                    if phi((k * control.h - alpha[N,2,0]) / control.h) == 0:
                        continue
                    # sum up all the particle stengths
                    w_sum0 = w_sum0 + (alpha[N,0,3] * phi((i * control.h - alpha[N,0,0]) / control.h) * phi((j * control.h - alpha[N,1,0]) / control.h) * phi((k * control.h - alpha[N,2,0]) / control.h))
                    w_sum1 = w_sum1 + (alpha[N,1,3] * phi((i * control.h - alpha[N,0,0]) / control.h) * phi((j * control.h - alpha[N,1,0]) / control.h) * phi((k * control.h - alpha[N,2,0]) / control.h))
                    w_sum2 = w_sum2 + (alpha[N,2,3] * phi((i * control.h - alpha[N,0,0]) / control.h) * phi((j * control.h - alpha[N,1,0]) / control.h) * phi((k * control.h - alpha[N,2,0]) / control.h))
                
                # assign the final vorticity values to the grid
                w[i,j,k,0] = w_sum0 / control.h ** 3
                w[i,j,k,1] = w_sum1 / control.h ** 3
                w[i,j,k,2] = w_sum2 / control.h ** 3

            # stop the timer
            end = timer()

            # print the current grid position, time/grid step, total time and predicted time
            print "\b" * 1000,
            print "z:(%2d/%2d), y:(%2d/%2d), time/step: %.2fs, time total: %.2fs/%.0fs"  %(k+1, control.N_grid,  j+1, control.N_grid, (end - start), (end - start1), (end - time1) + (((N_grid - k) * N_grid) + N_grid - j) * (end - start)),
    print "interpolating particles to grid done"


def inter_P2G_GPU(alpha,w,control): # interpolate the strengths of the particles onto the grid by using the GPU (not complete !!!)

    start1=timer()
    print "| 1/6) interpolating particles to grid              calc           |\r",

    # function kernel for the GPU
    mod = SourceModule("""
    #include <cmath>
    __device__ float phi(float x);
    __device__ int getGlobalIdx_3D_3D();
    __global__ void particle_strength(float* w, float* alpha, float h, int N_particles)
    {

        int i = threadIdx.x + blockDim.x * blockIdx.x;
        int j = threadIdx.y + blockDim.y * blockIdx.y;
        int k = threadIdx.z + blockDim.z * blockIdx.z;
        
        int idx_w = 3 * (k + (blockDim.z *  gridDim.z) * (j + (blockDim.y *  gridDim.y) * i));
        
        int idx_s0, idx_s1, idx_s2, idx_p0, idx_p1, idx_p2;
        
        float w_sum0 = 0;
        float w_sum1 = 0;
        float w_sum2 = 0;

        for(int N = 0 ; N < N_particles ; N++)
        {

            idx_s0 = 3 + 4 * (0 + 3 * N);
            idx_s1 = 3 + 4 * (1 + 3 * N);
            idx_s2 = 3 + 4 * (2 + 3 * N);
            
            idx_p0 = 0 + 4 * (0 + 3 * N);
            idx_p1 = 0 + 4 * (1 + 3 * N);
            idx_p2 = 0 + 4 * (2 + 3 * N);

            w_sum0 = w_sum0 + (alpha[idx_s0] * phi((i * h - alpha[idx_p0]) / h) * phi((j * h - alpha[idx_p1]) / h) * phi((k * h - alpha[idx_p2]) / h));
            w_sum1 = w_sum1 + (alpha[idx_s1] * phi((i * h - alpha[idx_p0]) / h) * phi((j * h - alpha[idx_p1]) / h) * phi((k * h - alpha[idx_p2]) / h));
            w_sum2 = w_sum2 + (alpha[idx_s2] * phi((i * h - alpha[idx_p0]) / h) * phi((j * h - alpha[idx_p1]) / h) * phi((k * h - alpha[idx_p2]) / h));
        
        }
        
        w[idx_w] = w_sum0 / pow(h,3);
        w[idx_w + 1] = w_sum1 / pow(h,3);
        w[idx_w + 2] = w_sum2 / pow(h,3);

    }

    __device__ float phi(float x)
    {
        float phi_x;

        if(abs(x) > 2)
        {
            return 0;
        }
        else if(abs(x) <= 1)
        {
            phi_x = (2 - 5 * pow(x,2) + 3 * pow(abs(x),3)) / 2;
            return phi_x;
        }
        else if(abs(x) > 1 && abs(x) <= 2)
        {
            phi_x = pow(2 - abs(x),2) * (1 - abs(x)) / 2;
            return phi_x;
        }
        return 0;
    }

    __device__ int getGlobalIdx_3D_3D()
    {
        int blockId = blockIdx.x + blockIdx.y * gridDim.x + gridDim.x * gridDim.y * blockIdx.z;
        int threadId = blockId * (blockDim.x * blockDim.y * blockDim.z) + (threadIdx.z * (blockDim.x * blockDim.y)) + (threadIdx.y * blockDim.x) + threadIdx.x;
        return threadId;
    }
    """)

    
    

    # allocate memory on the GPU and copy data to it
    w_32 = w.astype(np.float32)
    alpha_32 = alpha.astype(np.float32)
    
    w_gpu = drv.mem_alloc(w_32.nbytes)
    alpha_gpu = drv.mem_alloc(alpha_32.nbytes)

    drv.memcpy_htod(alpha_gpu, alpha_32)
    # get the function to be executed on the GPU
    particle_strength = mod.get_function("particle_strength")

    # convert variables to 32bit
    h_gpu = np.float32(control.h)
    N_particle_gpu = np.int32(control.N_particles)

    # specify block and grid dimensions
    blocksize = 8
    gridsize = control.N_grid/blocksize

    gpu_block = (blocksize, blocksize, blocksize)
    gpu_grid = (gridsize, gridsize, gridsize)

    # execute GPU-Kernel
    particle_strength(w_gpu, alpha_gpu, h_gpu, N_particle_gpu, block = gpu_block, grid = gpu_grid)

    # wait for all threads to finish
    pycuda.autoinit.context.synchronize()

    # copy the data from the GPU back to the program
    drv.memcpy_dtoh(w_32, w_gpu)

    end=timer()
    
    print "| 1/6) interpolating particles to grid              done   {0:6.2f}  |" .format(end-start1)
    return w_32
    
    

def vort_stream_equ(s,w,control):   # calculate the stream function field from the vorticity field on the grid points

    # solving the streamfunction equation with the SOR method, Dirichlet BC i.e. no flow on the walls
    w_SOR = 2/(1+np.sin(np.pi/control.N_grid))
    # start timer for the whole function
    time1=timer()

    SOR_iter = 100
    

    # iterate the solution for the stream function
    for iter in range(0,SOR_iter):

        
        # start timer for the current iteration
        time2=timer()

        # loop through all grid nodes except the outer layer (Dirichlet BC)
        for k in range(1,control.N_grid - 1):
            for j in range(1,control.N_grid - 1):
                for i in range(1,control.N_grid - 1):

                    s[i,j,k,0] = s[i,j,k,0] + w_SOR * ((s[i - 1,j,k,0] + s[i + 1,j,k,0] + s[i,j - 1,k,0] + s[i,j + 1,k,0] + s[i,j,k - 1,0] + s[i,j,k + 1,0] + control.h ** 2 * w[i,j,k,0]) / 6 - s[i,j,k,0])
                    s[i,j,k,1] = s[i,j,k,1] + w_SOR * ((s[i - 1,j,k,1] + s[i + 1,j,k,1] + s[i,j - 1,k,1] + s[i,j + 1,k,1] + s[i,j,k - 1,1] + s[i,j,k + 1,1] + control.h ** 2 * w[i,j,k,1]) / 6 - s[i,j,k,1])
                    s[i,j,k,2] = s[i,j,k,2] + w_SOR * ((s[i - 1,j,k,2] + s[i + 1,j,k,2] + s[i,j - 1,k,2] + s[i,j + 1,k,2] + s[i,j,k - 1,2] + s[i,j,k + 1,2] + control.h ** 2 * w[i,j,k,2]) / 6 - s[i,j,k,2])
        
        # stop the timer
        end=timer() 

        # print the current iteration, time/iteration, total time and predicted time
        print "\b" * 1000,
        print "iteration: %3d, time/iter: %.2fs, time total: %.2fs" %(iter+1, (end - time2), (end - time1)),
        iter = iter +1
    print "vorticity stream step done"
  

def vort_stream_equ_GPU(s,w,control):   # calculate the stream function field from the vorticity field on the grid points
    
    start1 = timer()

    print "| 2/6) vorticity-stream                             calc           |\r",

    mod = SourceModule("""
    #include <cmath>
    __global__ void vort_stream(float* w, float* s, float h, int N_grid)
    {
        int i = threadIdx.x + blockDim.x * blockIdx.x;
        int j = threadIdx.y + blockDim.y * blockIdx.y;
        int k = threadIdx.z + blockDim.z * blockIdx.z;
        
        int idx = 3 * (k + (blockDim.z *  gridDim.z) * (j + (blockDim.y *  gridDim.y) * i));

        if(i == 0 || j == 0 || k == 0 || i == N_grid-1 || j == N_grid-1 || k == N_grid-1)
        {
            return;
        }

        int idx_x1 = 3 * (k + (blockDim.z *  gridDim.z) * (j + (blockDim.y *  gridDim.y) * (i + 1)));
        int idx_x2 = 3 * (k + (blockDim.z *  gridDim.z) * (j + (blockDim.y *  gridDim.y) * (i - 1)));
        int idx_y1 = 3 * (k + (blockDim.z *  gridDim.z) * ((j + 1) + (blockDim.y *  gridDim.y) * i));
        int idx_y2 = 3 * (k + (blockDim.z *  gridDim.z) * ((j - 1) + (blockDim.y *  gridDim.y) * i));
        int idx_z1 = 3 * ((k + 1) + (blockDim.z *  gridDim.z) * (j + (blockDim.y *  gridDim.y) * i));
        int idx_z2 = 3 * ((k - 1) + (blockDim.z *  gridDim.z) * (j + (blockDim.y *  gridDim.y) * i));

        
        s[idx] = (pow(h,2) * w[idx] + s[idx_x1] + s[idx_x2] + s[idx_y1] + s[idx_y2] + s[idx_z1] + s[idx_z2]) / 6;
        s[idx + 1] = (pow(h,2) * w[idx + 1] + s[idx_x1 + 1] + s[idx_x2 + 1] + s[idx_y1 + 1] + s[idx_y2 + 1] + s[idx_z1 + 1] + s[idx_z2 + 1]) / 6;
        s[idx + 2] = (pow(h,2) * w[idx + 2] + s[idx_x1 + 2] + s[idx_x2 + 2] + s[idx_y1 + 2] + s[idx_y2 + 2] + s[idx_z1 + 2] + s[idx_z2 + 2]) / 6;
        
    }
    __global__ void VS_SOR_odd(float* w, float* s, float h, float w_SOR, int N_grid)
    {
        int i = threadIdx.x + blockDim.x * blockIdx.x;
        int j = threadIdx.y + blockDim.y * blockIdx.y;
        int k = threadIdx.z + blockDim.z * blockIdx.z;
        
        int idx = 3 * (k + (blockDim.z *  gridDim.z) * (j + (blockDim.y *  gridDim.y) * i));

        if(i == 0 || j == 0 || k == 0 || i == N_grid-1 || j == N_grid-1 || k == N_grid-1 || (i+j+k) % 2 == 0)
        {
            return;
        }

        int idx_x1 = 3 * (k + (blockDim.z *  gridDim.z) * (j + (blockDim.y *  gridDim.y) * (i + 1)));
        int idx_x2 = 3 * (k + (blockDim.z *  gridDim.z) * (j + (blockDim.y *  gridDim.y) * (i - 1)));
        int idx_y1 = 3 * (k + (blockDim.z *  gridDim.z) * ((j + 1) + (blockDim.y *  gridDim.y) * i));
        int idx_y2 = 3 * (k + (blockDim.z *  gridDim.z) * ((j - 1) + (blockDim.y *  gridDim.y) * i));
        int idx_z1 = 3 * ((k + 1) + (blockDim.z *  gridDim.z) * (j + (blockDim.y *  gridDim.y) * i));
        int idx_z2 = 3 * ((k - 1) + (blockDim.z *  gridDim.z) * (j + (blockDim.y *  gridDim.y) * i));

        
        s[idx] = s[idx] + w_SOR * ((pow(h,2) * w[idx] + s[idx_x1] + s[idx_x2] + s[idx_y1] + s[idx_y2] + s[idx_z1] + s[idx_z2]) / 6 - s[idx]);
        s[idx + 1] = s[idx + 1] + w_SOR * ((pow(h,2) * w[idx + 1] + s[idx_x1 + 1] + s[idx_x2 + 1] + s[idx_y1 + 1] + s[idx_y2 + 1] + s[idx_z1 + 1] + s[idx_z2 + 1]) / 6 - s[idx + 1]);
        s[idx + 2] = s[idx + 2] + w_SOR * ((pow(h,2) * w[idx + 2] + s[idx_x1 + 2] + s[idx_x2 + 2] + s[idx_y1 + 2] + s[idx_y2 + 2] + s[idx_z1 + 2] + s[idx_z2 + 2]) / 6 - s[idx + 2]);
    }
    __global__ void VS_SOR_even(float* w, float* s, float h, float w_SOR, int N_grid)
    {
        int i = threadIdx.x + blockDim.x * blockIdx.x;
        int j = threadIdx.y + blockDim.y * blockIdx.y;
        int k = threadIdx.z + blockDim.z * blockIdx.z;
        
        int idx = 3 * (k + (blockDim.z *  gridDim.z) * (j + (blockDim.y *  gridDim.y) * i));

        if(i == 0 || j == 0 || k == 0 || i == N_grid-1 || j == N_grid-1 || k == N_grid-1 || (i+j+k) % 2 == 1)
        {
            return;
        }

        int idx_x1 = 3 * (k + (blockDim.z *  gridDim.z) * (j + (blockDim.y *  gridDim.y) * (i + 1)));
        int idx_x2 = 3 * (k + (blockDim.z *  gridDim.z) * (j + (blockDim.y *  gridDim.y) * (i - 1)));
        int idx_y1 = 3 * (k + (blockDim.z *  gridDim.z) * ((j + 1) + (blockDim.y *  gridDim.y) * i));
        int idx_y2 = 3 * (k + (blockDim.z *  gridDim.z) * ((j - 1) + (blockDim.y *  gridDim.y) * i));
        int idx_z1 = 3 * ((k + 1) + (blockDim.z *  gridDim.z) * (j + (blockDim.y *  gridDim.y) * i));
        int idx_z2 = 3 * ((k - 1) + (blockDim.z *  gridDim.z) * (j + (blockDim.y *  gridDim.y) * i));

        
        s[idx] = s[idx] + w_SOR * ((pow(h,2) * w[idx] + s[idx_x1] + s[idx_x2] + s[idx_y1] + s[idx_y2] + s[idx_z1] + s[idx_z2]) / 6 - s[idx]);
        s[idx + 1] = s[idx + 1] + w_SOR * ((pow(h,2) * w[idx + 1] + s[idx_x1 + 1] + s[idx_x2 + 1] + s[idx_y1 + 1] + s[idx_y2 + 1] + s[idx_z1 + 1] + s[idx_z2 + 1]) / 6 - s[idx + 1]);
        s[idx + 2] = s[idx + 2] + w_SOR * ((pow(h,2) * w[idx + 2] + s[idx_x1 + 2] + s[idx_x2 + 2] + s[idx_y1 + 2] + s[idx_y2 + 2] + s[idx_z1 + 2] + s[idx_z2 + 2]) / 6 - s[idx + 2]);
    }
    """)


    # create 32-bit arrays for the GPU
    w_32 = w.astype(np.float32)
    s_32 = s.astype(np.float32)
    
    # allocate memory on the GPU
    w_gpu = drv.mem_alloc(w_32.nbytes)
    s_gpu = drv.mem_alloc(s_32.nbytes)

    # copy the data to the GPU
    drv.memcpy_htod(w_gpu, w_32)
    drv.memcpy_htod(s_gpu, s_32)

    # get the function to be executed on the GPU
    vort_stream = mod.get_function("vort_stream")
    VS_SOR_even = mod.get_function("VS_SOR_even")
    VS_SOR_odd = mod.get_function("VS_SOR_odd")

    # convert variables to 32bit
    h_gpu = np.float32(control.h)
    N_grid_gpu = np.int32(control.N_grid)

    w_SOR = 2/(1+np.sin(np.pi/control.N_grid))
    w_SOR_gpu = np.float32(w_SOR)

    # specify block and grid dimensions
    blocksize = 8
    gridsize = control.N_grid/blocksize

    gpu_block = (blocksize, blocksize, blocksize)
    gpu_grid = (gridsize, gridsize, gridsize)

    # number of iterations
    J_iter = 1000

    # loop
    for iter in range(0,J_iter):
        
        # execute GPU kernel
        #vort_stream(w_gpu, s_gpu, h_gpu, N_grid_gpu, block = gpu_block, grid = gpu_grid)
        VS_SOR_odd(w_gpu, s_gpu, h_gpu, w_SOR_gpu, N_grid_gpu, block = gpu_block, grid = gpu_grid)

        pycuda.autoinit.context.synchronize()

        VS_SOR_even(w_gpu, s_gpu, h_gpu, w_SOR_gpu, N_grid_gpu, block = gpu_block, grid = gpu_grid)

        pycuda.autoinit.context.synchronize()

    # copy the data from the GPU back to the program
    drv.memcpy_dtoh(s_32, s_gpu)
    
    end = timer()
    print "| 2/6) vorticity-stream                             done   {0:6.2f}  |" .format(end-start1)
    return s_32


def stream_velocity_equ(u,s,control):   # calculate the velocity field from the stream function field on the grid points
      
    # calculate the curl of the streamfunction to get the velocity field with the Central Difference Method
    
    # start timer for the whole function
    time1=timer()

    print "| 3/6) stream-velocity                              calc           |\r",
    
    # loop through all grid nodes except the outer layer ("Dirichlet BC")
    for k in range(1,control.N_grid - 1):
        
        #start the timer for z
        time2=timer()

        for j in range(1,control.N_grid - 1):
            for i in range(1,control.N_grid - 1):
                u[i,j,k,0] = ((s[i,j + 1,k,2] - s[i,j - 1,k,2]) - (s[i,j,k + 1,1] - s[i,j,k - 1,1])) / (2 * control.h)
                u[i,j,k,1] = ((s[i,j,k + 1,0] - s[i,j,k - 1,0]) - (s[i + 1,j,k,2] - s[i - 1,j,k,2])) / (2 * control.h)
                u[i,j,k,2] = ((s[i + 1,j,k,1] - s[i - 1,j,k,1]) - (s[i,j + 1,k,0] - s[i,j - 1,k,0])) / (2 * control.h)
        
        end=timer()
        print "\r",
        print "| 3/6) stream-velocity                            {0:3d}/{1:3d}  {2:>6.2f}  |\r" .format(k, control.N_grid-2, end-time1),
    
    print "| 3/6) stream-velocity                              done   {2:>6.2f}  |" .format(k, control.N_grid-2, end-time1)


def find_nearest(array, value): # find index of array with closest value to input

    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


def inter_G2P(alpha,u,control): # interpolate the velocity onto the particles

    start = timer()

    print "| 4/6) interpolating grid to particles              calc           |\r",

    # set up a linearly spaced "domain" to find the closest grid point to the particle
    dom = np.linspace(0,control.N_size,control.N_grid)
    
    # loop through all particles
    for N in range(0,control.N_particles):

        # find the nearest gridpoint
        idx_x = find_nearest(dom,alpha[N,0,0])
        idx_y = find_nearest(dom,alpha[N,1,0])
        idx_z = find_nearest(dom,alpha[N,2,0])
        
        # if closest particle is on the wall change to one next to it 
        if idx_x == 0 or idx_x == control.N_grid - 1 or idx_y == 0 or idx_y == control.N_grid - 1 or idx_z == 0 or idx_z == control.N_grid - 1:
            if idx_x == 0:
                idx_x = 1
            elif idx_x == control.N_grid - 1:
                idx_x = control.N_grid - 2
            if idx_y == 0:
                idx_y = 1
            elif idx_y == control.N_grid - 1:
                idx_y = control.N_grid - 2
            if idx_z == 0:
                idx_z = 1
            elif idx_z == control.N_grid - 1:
                idx_z = control.N_grid - 2
        
        # x-value for the interpolation
        interK_g_x = [dom[idx_x - 1],dom[idx_x],dom[idx_x + 1]]
        interK_g_y = [dom[idx_y - 1],dom[idx_y],dom[idx_y + 1]]
        interK_g_z = [dom[idx_z - 1],dom[idx_z],dom[idx_z + 1]]

        # u-value for the interpolation
        interK_u_x = [u[idx_x - 1,idx_y,idx_z],u[idx_x,idx_y,idx_z],u[idx_x + 1,idx_y,idx_z]]
        interK_u_y = [u[idx_x,idx_y - 1,idx_z],u[idx_x,idx_y,idx_z],u[idx_x,idx_y + 1,idx_z]]
        interK_u_z = [u[idx_x,idx_y,idx_z - 1],u[idx_x,idx_y,idx_z],u[idx_x,idx_y,idx_z - 1]]

        # calculate the polynomial (2nd order Lagrange)
        interL2_x = scipy.interpolate.lagrange(interK_g_x, interK_u_x)
        interL2_y = scipy.interpolate.lagrange(interK_g_y, interK_u_y)
        interL2_z = scipy.interpolate.lagrange(interK_g_z, interK_u_z)

        # calculate the first derivate of the polynomial (needed for stretching)
        interL2_dx = np.polyder(interL2_x,1)
        interL2_dy = np.polyder(interL2_y,1)
        interL2_dz = np.polyder(interL2_z,1)

        #calculate the veloctiy value at the position of the particle
        alpha[N,0,1] = interL2_x(alpha[N,0,0])
        alpha[N,1,1] = interL2_y(alpha[N,1,0])
        alpha[N,2,1] = interL2_z(alpha[N,2,0])

        #calculate the acceleration value at the position of the particle
        alpha[N,0,2] = interL2_dx(alpha[N,0,0])
        alpha[N,1,2] = interL2_dy(alpha[N,1,0])
        alpha[N,2,2] = interL2_dz(alpha[N,2,0])
    
    end = timer()

    print "| 4/6) interpolating grid to particles              done   {0:6.2f}  |" .format(end-start)


def move_part(alpha,control):   # move the particles i.e time advancement (maybe use RK4?!?!)

    start = timer()

    print "| 5/6) move particles                               calc           |\r",

    # loop through all particles
    for N in range(0,control.N_particles):

        # calculate new positions
        alpha[N,0,0] = alpha[N,0,0] + alpha[N,0,1] * control.t_step
        alpha[N,1,0] = alpha[N,1,0] + alpha[N,1,1] * control.t_step
        alpha[N,2,0] = alpha[N,2,0] + alpha[N,2,1] * control.t_step
    
    end = timer()

    print "| 5/6) move particles                               done   {0:6.2f}  |" .format(end-start)
    

def vel_RK4(control,val_0,fun): # may be needed for time advancement (not complete!!!)

    # allocate solution array
    val_1=np.zeros(3)

    # integrate the value using Runge-Kutta 4
    k1_0 = fun(val_0[0])
    k1_1 = fun(val_0[1])
    k1_2 = fun(val_0[2])

    k2_0 = val_0[0] + control.t_step / 2 * k1_0
    k2_1 = val_0[1] + control.t_step / 2 * k1_1
    k2_2 = val_0[2] + control.t_step / 2 * k1_2

    k3_0 = val_0[0] + control.t_step / 2 * k2_0
    k3_1 = val_0[1] + control.t_step / 2 * k2_1
    k3_2 = val_0[2] + control.t_step / 2 * k2_2

    k4_0 = val_0[0] + control.t_step * k3_0
    k4_1 = val_0[1] + control.t_step * k3_1
    k4_2 = val_0[2] + control.t_step * k3_2

    val_1[0] = val_0[0] + control.t_step / 6 * (k1_0 + 2 * k2_0 + 2 * k3_0 + k4_0)
    val_1[1] = val_0[1] + control.t_step / 6 * (k1_1 + 2 * k2_1 + 2 * k3_1 + k4_1)
    val_1[2] = val_0[2] + control.t_step / 6 * (k1_2 + 2 * k2_2 + 2 * k3_2 + k4_2)

    return val_1


def stretching(alpha, control): # strechting of the vortex particle strengths due to three dimensionality

    start = timer()

    print "| 6/6) stretching                                   calc           |\r",
     # loop through all particles
    for N in range(0,control.N_particles):

        # calculate stretching effect i.e.  update particle strentghs
        alpha[N,0,3] = alpha[N,0,3] + control.t_step * alpha[N,0,3] * alpha[N,0,2]
        alpha[N,1,3] = alpha[N,1,3] + control.t_step * alpha[N,1,3] * alpha[N,1,2]
        alpha[N,2,3] = alpha[N,2,3] + control.t_step * alpha[N,2,3] * alpha[N,2,2]

    end = timer()

    print "| 6/6) stretching                                   done   {0:6.2f}  |" .format(end-start)


def remesh_part(): # still to be coded but probably not needed

    return 0


def save_values(save_file,w,s,u,control): # save the current values uf vorticity field, stream function field and velocity field (not complete!!!)

    np.save(save_file,w)
    np.save(save_file,s)
    np.save(save_file,u)

    return 0


def create_savefile(): # create a file to save the simulation data to (not complete!!!)

    curr_dt = datetime.datetime.now()
    name = "VIC_Sim_results_" + str(curr_dt.year) + str(curr_dt.month) + str(curr_dt.day) + ".txt"
    save_file = open(name,"w+")
    
    return save_file


def create_plot(alpha,control): # plot the current paticle positions in the domain

    mpl.interactive(True)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(alpha[:,0,0], alpha[:,1,0], alpha[:,2,0], c='r', marker='.')
    ax.set_xlim3d(0,control.N_size)
    ax.set_ylim3d(0,control.N_size)
    ax.set_zlim3d(0,control.N_size)
    plt.draw()
    plt.pause(0.01)
    return fig
    

def update_plot(fig,alpha,control):

    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(alpha[:,0,0], alpha[:,1,0], alpha[:,2,0], c='r', marker='.')
    ax.set_xlim3d(0,control.N_size)
    ax.set_ylim3d(0,control.N_size)
    ax.set_zlim3d(0,control.N_size)
    plt.draw()
    plt.pause(0.01)
    


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


def print_header(control):
    print "+------------------------------------------------------------------+"
    print "| Particles = {0:5d} | Grid = {1:3d} | Size = {2:2.2f} | time = {3:0.2f}:{4:2.2f} |" .format(control.N_particles, control.N_grid, control.N_size, control.t_step, control.t_total)
    print "+------------------------------------------------------------------+"
    print "| current time step: {0:0.2f}                          status    time  |" .format(control.t_time)


def print_foot(control,t,w,s,u):
    print "|                                                complete  {0:6.0f}  |".format(t)
    print "+------------------------------------------------------------------+"
    print "|non-zero values:                                                  |"
    print "|w:{0:8d}, s:{1:8d}, u:{2:8d} / {3:8d}                     |" .format(count_nonzero(w), count_nonzero(s), count_nonzero(u), N_grid ** 3 * 3)
    print "+------------------------------------------------------------------+\n\n"
#################################   MAIN - FUNCTION   #################################


# set simulation parameters
N_size = 2 * np.pi
N_rings = 1
N_grid = 64
N_particles = 1250
h = N_size / (N_grid - 1)
t_total = 10
t_step = 0.01
t_time = t_step
kin_vis = 0.001

# save simulation parameters in ctrl class
control = ctrl(N_size, N_grid, N_particles, h, t_step, t_total, t_time, kin_vis)

# set and save parameters for one ring in ring_val class
ring1 = ring_val(1.5, 0.3, np.pi, np.pi, np.pi, 100, 5, 50, 5)

# initialize the grid fields (each value is a 3-vector)
w = np.zeros((N_grid,N_grid,N_grid,3))      # vorticity field
s = np.zeros((N_grid,N_grid,N_grid,3))      # stream function field
u = np.zeros((N_grid,N_grid,N_grid,3))      # velocity field

# initialize the particle array (each particle has 4 values (position:[0], velocity:[1], acceleration:[2], particle/vortex strength:[3]); each value is a 3-vector)
alpha = np.zeros((N_particles,3,4))

# create a vortex ring and plot it
c_ring2(ring1,alpha,control)

fig = create_plot(alpha,control)

#save_file=create_savefile()

# counting variable for plotting the particles in intervals
count_print = 1

#################################   simulation loop   #################################

print "\nSimulation start \n"

# main loop
while control.t_time < control.t_total:

    # start timer for the current simulation step
    time1=timer()

    # print time step
    print_header(control)

    # VIC-algorithm
    w = inter_P2G_GPU(alpha,w,control)
    s = vort_stream_equ_GPU(s,w,control)
    stream_velocity_equ(u,s,control)
    inter_G2P(alpha,u,control)
    move_part(alpha,control)
    stretching(alpha, control)

    # stop the timer
    end=timer()

    # save the values (not completely done yet !!!)
    # save_values(save_file,w,s,u,control)

    print_foot(control,(end-time1),w,s,u)
    # update plot
    if count_print % 1 == 0:
        update_plot(fig,alpha,control)

    count_print = count_print+1
    control.t_time = control.t_time + control.t_step
