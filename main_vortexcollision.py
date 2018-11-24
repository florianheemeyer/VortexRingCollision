import control
import ring_val
import interpolate
import numerical
import numpy as np
from timeit import default_timer as timer



if __name__ == '__main__':
    # set simulation parameters
    N_size = 2 * np.pi
    N_rings = 1
    N_grid = 16
    N_particles = 1200
    h = N_size / (N_grid - 1)
    t_total = 10
    t_step = 0.01
    t_time = t_step
    kin_vis = 0.01

    # save simulation parameters in ctrl class
    control = control.ctrl(N_size, N_grid, N_particles, h, t_step, t_total, t_time, kin_vis)
    # set and save parameters for one ring in ring_val class
    ring1 = ring_val.ring_val(1.5, 0.3, np.pi, np.pi, np.pi, 100, 8, 30, 5)

    # initialize the grid fields (each value is a 3-vector)
    w = np.zeros((N_grid, N_grid, N_grid, 3))  # vorticity field
    s = np.zeros((N_grid, N_grid, N_grid, 3))  # stream function field
    u = np.zeros((N_grid, N_grid, N_grid, 3))  # velocity field

    # initialize the particle array (each particle has 4 values (position:[0], velocity:[1], acceleration:[2], particle/vortex strength:[3]); each value is a 3-vector)
    alpha = np.zeros((N_particles, 3, 4))

    # create a vortex ring and plot it
    ring_val.c_ring2(ring1, alpha, control)
    ring_val.plot_particles(alpha, control)

    # save_file=create_savefile()

    # counting variable for ploting the particles in intervals
    count = 1

    #################################   simulation loop   #################################

    print "\nSimulation start \n"
    # main loop
    while control.t_time < control.t_total:
        # start timer for the current simulation step
        time1 = timer()

        # print  time step
        print "time_step %.2f" % control.t_time

        # VIC-algorithm
        interpolate.inter_P2G(alpha, w, control)
        numerical.vort_stream_equ(s, w, control)
        numerical.stream_velocity_equ(u, s, control)
        interpolate.inter_G2P(alpha, u, control)
        ring_val.move_part(alpha, control)
        ring_val.stretching(alpha, control)

        # stop the timer
        end = timer()

        # if statement is true plot the particles (or comment out)
        # if count_print % 10 == 0:
        ring_val.plot_particles(alpha, control)

        # save the values (not completely done yet !!!)
        # save_values(save_file,w,s,u,control)

        # print some time step information for quality control
        print "non-zero w: %d/%d, non-zero s: %d/%d,non-zero u %d/%d" % (
        count_nonzero(w), N_grid ** 3, count_nonzero(s), N_grid ** 3, count_nonzero(u), N_grid ** 3)
        print "\n%.2fs for time step %.2f\n\n" % ((end - time1), control.t_time)
        count = count + 1
        control.t_time = control.t_time + control.t_step








