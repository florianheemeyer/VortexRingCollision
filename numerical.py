from timeit import default_timer as timer


def vort_stream_equ(s, w, control):  # calculate the stream function field from the vorticity field on the grid points

    # solving the streamfunction equation with the SOR method, Dirichlet BC i.e. no flow on the walls
    w_SOR = 1.5
    SOR_iter = 500

    # start timer for the whole function
    time1 = timer()

    # iterate the solution for the stream function
    for iter in range(0, SOR_iter):

        # start timer for the current iteration
        time2 = timer()

        # loop through all grid nodes except the outer layer (Dirichlet BC)
        for k in range(1, control.N_grid - 1):
            for j in range(1, control.N_grid - 1):
                for i in range(1, control.N_grid - 1):
                    s[i, j, k, 0] = s[i, j, k, 0] + w_SOR * ((1 / 6 * (
                                s[i - 1, j, k, 0] + s[i + 1, j, k, 0] + s[i, j - 1, k, 0] + s[i, j + 1, k, 0] + s[
                            i, j, k - 1, 0] + s[i, j, k + 1, 0]) + control.h ** 3 * w[i, j, k, 0]) - s[i, j, k, 0])
                    s[i, j, k, 1] = s[i, j, k, 1] + w_SOR * ((1 / 6 * (
                                s[i - 1, j, k, 1] + s[i + 1, j, k, 1] + s[i, j - 1, k, 1] + s[i, j + 1, k, 1] + s[
                            i, j, k - 1, 1] + s[i, j, k + 1, 1]) + control.h ** 3 * w[i, j, k, 1]) - s[i, j, k, 1])
                    s[i, j, k, 2] = s[i, j, k, 2] + w_SOR * ((1 / 6 * (
                                s[i - 1, j, k, 2] + s[i + 1, j, k, 2] + s[i, j - 1, k, 2] + s[i, j + 1, k, 2] + s[
                            i, j, k - 1, 2] + s[i, j, k + 1, 2]) + control.h ** 3 * w[i, j, k, 2]) - s[i, j, k, 2])

        # stop the timer
        end = timer()

        # print the current iteration, time/iteration, total time and predicted time
        #print "\b" * 1000,
        #print "iteration: %3d/%3d, time/iter: %.2fs, time total: %.2fs/%.0fs" % (
        #iter + 1, SOR_iter, (end - time2), (end - time1), (end - time1) + (SOR_iter - iter + 1) * (end - time2)),
    print "vorticity stream step done"


def stream_velocity_equ(u, s, control):  # calculate the velocity field from the stream function field on the grid points

    # calculate the curl of the streamfunction to get the velocity field with the Central Difference Method

    # loop through all grid nodes except the outer layer ("Dirichlet BC")
    for k in range(1, control.N_grid - 1):
        for j in range(1, control.N_grid - 1):
            for i in range(1, control.N_grid - 1):
                u[i, j, k, 0] = ((s[i, j + 1, k, 2] - s[i, j - 1, k, 2]) - (s[i, j, k + 1, 1] - s[i, j, k - 1, 1])) / (
                            2 * control.h)
                u[i, j, k, 1] = ((s[i, j, k + 1, 0] - s[i, j, k - 1, 0]) - (s[i + 1, j, k, 2] - s[i - 1, j, k, 2])) / (
                            2 * control.h)
                u[i, j, k, 2] = ((s[i + 1, j, k, 1] - s[i - 1, j, k, 1]) - (s[i, j + 1, k, 0] - s[i, j - 1, k, 0])) / (
                            2 * control.h)

    print "stream-velocity step done"

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
