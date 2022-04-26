import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation

def aninmate(animate_func, frames, intervals):
    # Plotting the Animation
    fig = plt.figure()
    ax = plt.axes()
    ani = animation.FuncAnimation(
        fig,
        animate_func, interval=intervals,   
                                    frames=frames)
    plt.show()