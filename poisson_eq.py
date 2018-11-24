"""
This file is to test the poisson equation
"""
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np

def compute_poisson():
    # Initialize figure
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    # Set parameters for testing
    # These parameters can usally obtained from control instance
    h = 0.1
    size = 100
    Z = np.zeros((size, size)) # Z is in this case the stream matrix
    '''Random boundary condtions for testing'''
    Z[:,0] = 10.
    Z[0,:] = 5.
    Z[:,-1] = 5.
    '''Make mesh for 3D-plot'''
    X = np.arange(-size/2, size/2, 1)
    Y = np.arange(-size/2, size/2, 1)
    X, Y = np.meshgrid(X, Y)
    '''counting variable'''
    k = 0
    error = 1
    '''Calculate new Z in a iterative matter until desired error has been reached'''
    while k < 1000:
        Z_old = Z
        #print Z_old
        k += 1
        for i in range(1,size-1):
            for j in range(1,size-1):
                Z[i,j] = (1./4.)* (Z[i+1,j] + Z[i-1,j] + Z[i,j+1] + Z[i,j-1])

    # Plot the surface.
    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)

    # Customize the z axis.
    ax.set_zlim(0, 15)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)

    plt.show()
    return Z




if __name__ == '__main__':
    S = compute_poisson()

