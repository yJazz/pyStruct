from scipy.interpolate import griddata
import numpy as np


def plot_cartesian(fig, ax, y,z,value, resolution = 50,contour_method='cubic', vmin=-0.002, vmax=0.002):
    resolution = str(resolution)+'j'
    X,Y = np.mgrid[min(y):max(y):complex(resolution),   min(z):max(z):complex(resolution)]
    points = [[a,b] for a,b in zip(y,z)]
    Z = griddata(points, value, (X, Y), method=contour_method)
    cs = ax.contourf(X,Y,Z, cmap='jet', vmin=vmin, vmax=vmax)
    return cs