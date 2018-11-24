import numpy as np
import scipy as sp
import scipy.interpolate
import matplotlib.pylab as plt
from mpl_toolkits.mplot3d import axes3d
from sys import getsizeof
import datetime
import time
from timeit import default_timer as timer

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



def c_ring(R, r, w_0, Re, alpha, control):
    # create counting variables and poistion of the ring
    count_ring = 0
    N = 0
    x_pos = np.pi
    y_pos = np.pi
    z_pos = np.pi
    Re = 1000

    # loop through all the grid nodes
    for k in range(0, control.N_grid):
        for j in range(0, control.N_grid):
            for i in range(0, control.N_grid):

                # assign position to the particles
                alpha[N, 0, 0] = i * control.h
                alpha[N, 1, 0] = j * control.h
                alpha[N, 2, 0] = k * control.h

                # condition for particle to gain initial vorticity
                if (((i * control.h - x_pos) ** 2 + (j * control.h - y_pos) ** 2) ** 0.5 - R) ** 2 + (
                        k * control.h - z_pos) ** 2 < r ** 2:
                    # calculate some values needed for vorticity
                    gam = Re * control.kin_vis
                    rad = ((i * control.h) ** 2 + (j * control.h) ** 2) ** 0.5
                    rho = ((rad - R) ** 2 + (k * control.h) ** 2) ** 0.5
                    th = np.arctan2((j * control.h), (i * control.h))

                    # calculate magnitude of vorticity
                    w_mag = gam / (np.pi * R ** 2) * np.exp(-(rho / r) ** 2)

                    # vectorize vorticity
                    alpha[N, 0, 1] = np.sin(th) * w_mag
                    alpha[N, 1, 1] = -np.cos(th) * w_mag
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
    plt.show()"""  #### probably not needed ###


def c_ring2(ring, alpha, control):  # create the particles for one vortex ring given the parameters passed with "ring"

    # create counting and loop variables
    N = 0
    th = np.linspace(0, 2 * np.pi, ring.N_th + 1)
    phi = np.linspace(0, 2 * np.pi, ring.N_phi + 1)
    d = np.linspace(ring.r / 10, ring.r, ring.N_d)

    # loop through all positions in the ring
    for k in range(0, ring.N_d):
        for j in range(0, ring.N_phi):
            for i in range(0, ring.N_th):
                # assign positions to the particles
                alpha[N, 0, 0] = ring.x_pos + (ring.R + d[k] * np.cos(th[i])) * np.cos(phi[j])
                alpha[N, 1, 0] = ring.y_pos + (ring.R + d[k] * np.cos(th[i])) * np.sin(phi[j])
                alpha[N, 2, 0] = ring.z_pos + d[k] * np.sin(th[i])

                # calculate the mangitude of vorticity
                gam = ring.Re * control.kin_vis
                w_mag = gam / (np.pi * ring.R ** 2) * np.exp(-(d[k] / ring.r) ** 2)

                # vectorize vorticity
                alpha[N, 0, 3] = np.sin(th[i]) * w_mag
                alpha[N, 1, 3] = -np.cos(th[i]) * w_mag
                N = N + 1

                # print out counting variable
    print "%d particles created" % N


def plot_particles(alpha,control): # plot the current paticle positions in the domain

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(alpha[:,0,0], alpha[:,1,0], alpha[:,2,0], c='r', marker='.')
    ax.set_xlim3d(0,control.N_size)
    ax.set_ylim3d(0,control.N_size)
    ax.set_zlim3d(0,control.N_size)
    plt.show()


def stretching(alpha, control):  # strechting of the vortex particle strengths due to three dimensionality

    # loop through all particles
    for N in range(0, control.N_particles):
        # calculate stretching effect i.e.  update particle strentghs
        alpha[N, 0, 3] = alpha[N, 0, 3] + control.t_step * alpha[N, 0, 3] * alpha[N, 0, 2]
        alpha[N, 1, 3] = alpha[N, 1, 3] + control.t_step * alpha[N, 1, 3] * alpha[N, 1, 2]
        alpha[N, 2, 3] = alpha[N, 2, 3] + control.t_step * alpha[N, 2, 3] * alpha[N, 2, 2]

    print "stretching step done"


def move_part(alpha,control):   # move the particles i.e time advancement (maybe use RK4?!?!)

    # loop through all particles
    for N in range(0,control.N_particles):

        # calculate new positions
        alpha[N,0,0] = alpha[N,0,0] + alpha[N,0,1] * control.t_step
        alpha[N,1,0] = alpha[N,1,0] + alpha[N,1,1] * control.t_step
        alpha[N,2,0] = alpha[N,2,0] + alpha[N,2,1] * control.t_step

    print "moving step done"