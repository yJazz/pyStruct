from scipy.interpolate import griddata
import numpy as np
import pandas as pd


def plot_cartesian(fig, ax, y,z,value, resolution = 50,contour_method='cubic', vmin=-0.002, vmax=0.002):
    resolution = str(resolution)+'j'
    X,Y = np.mgrid[min(y):max(y):complex(resolution),   min(z):max(z):complex(resolution)]
    points = [[a,b] for a,b in zip(y,z)]
    Z = griddata(points, value, (X, Y), method=contour_method)
    cs = ax.contourf(X,Y,Z, cmap='jet', vmin=vmin, vmax=vmax)
    return cs



class PolarContourPlot:
    def __init__(
        self, 
        df: pd.DataFrame,
        cmap = 'jet', 
    ):
        assert type(df) == pd.DataFrame
        assert list(df.columns[:3]) == ['x', 'y', 'z']
        self.df = df
        self.cmap = cmap
        # 
        self.columns = self.df.columns[3:]
        self.N_grids = len(df)
        self.R_resolution = 10
        self.theta_resolution = 20
        self.level = 20
    

    def plot(self, cmap='jet', title='', figsize=None, show=True):
        if not figsize:
            figsize = (4*len(self.columns), 4 )
        fig, axes = plt.subplots(
            nrows=1, ncols=len(self.columns), 
            subplot_kw={'projection': 'polar'},
            figsize=figsize
            )
        self.radius = self.df['z'].max()

        if len(self.columns) == 1:
            ax = axes
            ax = self._plot_single(fig, ax, self.columns[0])
        else:
            for i, col in enumerate(self.columns):
                ax = axes[i]
                ax = self._plot_single(fig, ax, col)
        
        fig.suptitle(title, fontweight='semibold', y=1.1)
        plt.tight_layout()
        if show:
            plt.show()
        else:
            plt.close()
        return fig, ax


    def _plot_single(self, fig, ax, col):
        # Plot init
        ax.set_title(col)
        ax.set_rticks(self.radius * 0.1 * np.arange(2, 12, 2))
        ax.set_yticklabels(['' for i in range(5)])
        ax.set_xticks(ax.get_xticks().tolist()) 
        ax.set_xticklabels(['$90^o$', '$45^o$', '$0^o$', '$315^o$', '$270^o$', '$225^o$', '$180^o$', '$135^o$'])

        # Get data
        Theta, R, output_matrix = self._map_data(col)
        # Plot
        cs = ax.contourf(Theta,
                         R,
                         output_matrix,
                         self.level,
                         cmap=self.cmap,
                         alpha=0.8)

        # Color bar
        cbar = fig.colorbar(cs, ax=ax, orientation="horizontal", format="%.2f")
        cbar.ax.locator_params(nbins=7)
        return fig, ax, cs

    def set_resolution(self, R_resolution, theta_resolution, level):
        self.R_reolution = R_resolution
        self.theta_resolution = theta_resolution
        self.level = level

    def _find_closest_point(self, file, x, y, z):
        x_table = file['x']
        y_table = file['y']
        z_table = file['z']
        distance = np.sqrt((x - x_table) ** 2 + (y - y_table) ** 2 + (z - z_table) ** 2)
        file['distance'] = distance
        representative_point = file['distance'].argmin()
        representative_point_col = file.iloc[representative_point]
        return representative_point_col

    def _map_data(self, column_name):
        R = np.linspace(0, self.radius, self.R_resolution)
        Theta = np.linspace(0, 2 * np.pi, self.theta_resolution)
        output_matrix = np.zeros((self.R_resolution, self.theta_resolution))

        x = self.df['x']
        for j, theta in enumerate(Theta):
            for i, r in enumerate(R):
                y = -r * np.cos(theta)
                z = r * np.sin(theta)
                col = self._find_closest_point(self.df, x,  y, z)
                output_matrix[i][j] = col[column_name]
        return Theta, R, output_matrix

        

