import numpy as np
import scipy as sp
import scipy.interpolate
import matplotlib.pylab as plt
from mpl_toolkits.mplot3d import axes3d
from sys import getsizeof
import datetime
import time
from timeit import default_timer as timer
import main_vortexcollision
#import pycuda as cuda
#import pycuda.gpuarray as gpuarray
#import pycuda.autoinit
#from pycuda.compiler import SourceModule

def inter_P2G(alpha, w, control):  # interpolate the strengths of the particles onto the grid

    # start the timer for the whole function
    start1 = timer()

    # loop through all the grid nodes
    for k in range(0, control.N_grid):
        for j in range(0, control.N_grid):

            # start the timer for the current yz-grid step
            start = timer()
            for i in range(0, control.N_grid):

                # set the summation variable to 0
                w_sum0 = 0
                w_sum1 = 0
                w_sum2 = 0

                # loop through all particle positions
                for N in range(0, control.N_particles):

                    # check if one of the interpolations is zero. If yes jump to the next particle.
                    if phi((i * control.h - alpha[N, 0, 0]) / control.h) == 0:
                        continue
                    if phi((j * control.h - alpha[N, 1, 0]) / control.h) == 0:
                        continue
                    if phi((k * control.h - alpha[N, 2, 0]) / control.h) == 0:
                        continue
                    # sum up all the particle stengths
                    w_sum0 = w_sum0 + (alpha[N, 0, 3] * phi((i * control.h - alpha[N, 0, 0]) / control.h) * phi(
                        (j * control.h - alpha[N, 1, 0]) / control.h) * phi(
                        (k * control.h - alpha[N, 2, 0]) / control.h))
                    w_sum1 = w_sum1 + (alpha[N, 1, 3] * phi((i * control.h - alpha[N, 0, 0]) / control.h) * phi(
                        (j * control.h - alpha[N, 1, 0]) / control.h) * phi(
                        (k * control.h - alpha[N, 2, 0]) / control.h))
                    w_sum2 = w_sum2 + (alpha[N, 2, 3] * phi((i * control.h - alpha[N, 0, 0]) / control.h) * phi(
                        (j * control.h - alpha[N, 1, 0]) / control.h) * phi(
                        (k * control.h - alpha[N, 2, 0]) / control.h))

                # assign the final vorticity values to the grid
                w[i, j, k, 0] = w_sum0 / control.h ** 3
                w[i, j, k, 1] = w_sum1 / control.h ** 3
                w[i, j, k, 2] = w_sum2 / control.h ** 3

            # stop the timer
            end = timer()

            # print the current grid position, time/grid step, total time and predicted time
            print "\b" * 1000,
            #print "z:(%2d/%2d), y:(%2d/%2d), time/step: %.2fs, time total: %.2fs/%.0fs" % (
            #k + 1, control.N_grid, j + 1, control.N_grid, (end - start), (end - start1),
            #(end - main_vortexcollision.time1) + (((N_grid - k) * N_grid) + N_grid - j) * (end - start)),
    print "interpolating particles to grid done"


def inter_P2G_GPU(alpha, w,
                  control):  # interpolate the strengths of the particles onto the grid by using the GPU (not complete !!!)

    # allocate memory on the GPU
    w_gpu = gpuarray.to_gpu(w)
    alpha_gpu = gpuarray.to_gpu(alpha)

    # get the function to be executed on the GPU
    particle_strength = mod.get_function("particle_strength")
    particle_strength.prepare([numpy.intp, numpy.intp], block=(nthreads_x, nthreads_x, nthreads_x))
    for k in range(0, control.N_grid):
        # place CUDA kernel here that does
        """
        for j in range(0,control.N_grid):
            for i in range(0,control.N_grid):

                # set the summation variable to 0
                w_sum0 = 0
                w_sum1 = 0
                w_sum2 = 0

                # loop through all particle positions
                for N in range(0,control.N_particles):

                    # sum up all the particle stengths
                    w_sum0 = w_sum0 + (alpha[N,0,3] * phi((i * control.h - alpha[N,0,0]) / control.h) * phi((j * control.h - alpha[N,1,0]) / control.h) * phi((k * control.h - alpha[N,2,0]) / control.h))
                    w_sum1 = w_sum1 + (alpha[N,1,3] * phi((i * control.h - alpha[N,0,0]) / control.h) * phi((j * control.h - alpha[N,1,0]) / control.h) * phi((k * control.h - alpha[N,2,0]) / control.h))
                    w_sum2 = w_sum2 + (alpha[N,2,3] * phi((i * control.h - alpha[N,0,0]) / control.h) * phi((j * control.h - alpha[N,1,0]) / control.h) * phi((k * control.h - alpha[N,2,0]) / control.h))

                # assign the final vorticity values to the grid
                w[i,j,k,0] = w_sum0 / control.h ** 3
                w[i,j,k,1] = w_sum1 / control.h ** 3
                w[i,j,k,2] = w_sum2 / control.h ** 3"""

    print "interpolating particles to grid done"

def phi(x):
    # interpolation function M4'-Spline
    if abs(x) > 2:
        return 0
    elif abs(x) <= 1:
        phi_x = (2 - 5 * pow(x, 2) + 3 * pow(abs(x), 3)) / 2
    elif abs(x) > 1 and abs(x) <= 2:
        phi_x = pow(2 - abs(x), 2) * (1 - abs(x)) / 2

    return phi_x

def find_nearest(array, value): # find index of array with closest value to input

    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


def inter_G2P(alpha, u, control):  # interpolate the velocity onto the particles

    # set up a linearly spaced "domain" to find the closest grid point to the particle
    dom = np.linspace(0, control.N_size, control.N_grid)

    # loop through all particles
    for N in range(0, control.N_particles):

        # find the nearest gridpoint
        idx_x = find_nearest(dom, alpha[N, 0, 0])
        idx_y = find_nearest(dom, alpha[N, 1, 0])
        idx_z = find_nearest(dom, alpha[N, 2, 0])

        # if closest particle is on the wall change to one next to it
        if idx_x == 0 or idx_x == control.N_grid or idx_y == 0 or idx_y == control.N_grid or idx_z == 0 or idx_z == control.N_grid:
            if idx_x == 0:
                idx_x = 1
            elif idx_x == control.N_grid - 1:
                idx_x = control.N_grid - 2
            elif idx_y == 0:
                idx_y = 1
            elif idx_y == control.N_grid - 1:
                idx_y = control.N_grid - 2
            elif idx_z == 0:
                idx_z = 1
            elif idx_z == control.N_grid - 1:
                idx_z = control.N_grid - 2

        # x-value for the interpolation
        interK_g_x = [dom[idx_x - 1], dom[idx_x], dom[idx_x + 1]]
        interK_g_y = [dom[idx_y - 1], dom[idx_y], dom[idx_y + 1]]
        interK_g_z = [dom[idx_z - 1], dom[idx_z], dom[idx_z + 1]]

        # u-value for the interpolation
        interK_u_x = [u[idx_x - 1, idx_y, idx_z], u[idx_x, idx_y, idx_z], u[idx_x + 1, idx_y, idx_z]]
        interK_u_y = [u[idx_x, idx_y - 1, idx_z], u[idx_x, idx_y, idx_z], u[idx_x, idx_y + 1, idx_z]]
        interK_u_z = [u[idx_x, idx_y, idx_z - 1], u[idx_x, idx_y, idx_z], u[idx_x, idx_y, idx_z - 1]]

        # calculate the polynomial (2nd order Lagrange)
        interL2_x = scipy.interpolate.lagrange(interK_g_x, interK_u_x)
        interL2_y = scipy.interpolate.lagrange(interK_g_y, interK_u_y)
        interL2_z = scipy.interpolate.lagrange(interK_g_z, interK_u_z)

        # calculate the first derivate of the polynomial (needed for stretching)
        interL2_dx = np.polyder(interL2_x, 1)
        interL2_dy = np.polyder(interL2_y, 1)
        interL2_dz = np.polyder(interL2_z, 1)

        # calculate the veloctiy value at the position of the particle
        alpha[N, 0, 1] = interL2_x(alpha[N, 0, 0])
        alpha[N, 1, 1] = interL2_y(alpha[N, 1, 0])
        alpha[N, 2, 1] = interL2_z(alpha[N, 2, 0])

        # calculate the acceleration value at the position of the particle
        alpha[N, 0, 2] = interL2_dx(alpha[N, 0, 0])
        alpha[N, 1, 2] = interL2_dy(alpha[N, 1, 0])
        alpha[N, 2, 2] = interL2_dz(alpha[N, 2, 0])

    print "interpolating grid to particle done"