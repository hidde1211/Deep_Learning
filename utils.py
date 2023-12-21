import numpy as np
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt

def get_grid_world(start_point,end_point,dx,step):
    dLat = end_point['coordinates'][0] - start_point['coordinates'][0]
    dLon = end_point['coordinates'][1] - start_point['coordinates'][1]
    initial_heading = (360 + np.rad2deg(np.arctan2(dLon, dLat))) % 360

    lat_grid = np.arange(35, 60, step)
    lon_grid = np.arange(-75, 3, step)

    # Calculate the number of rows and columns based on the step size
    grid_width = len(lon_grid)
    grid_height = len(lat_grid)

    return initial_heading,lat_grid,lon_grid,grid_width,grid_height

def get_heading_to_end_point(s,end_point):
    dLat = end_point['coordinates'][0] - s.lat
    dLon = end_point['coordinates'][1] - s.lon
    initial_heading = (360 + np.rad2deg(np.arctan2(dLon, dLat))) % 360
    return initial_heading

def get_figure(start_point, end_point, lat, lon, U, V):
    fig = plt.figure(figsize=(15, 5))
    gs = gridspec.GridSpec(3, 2)
    ax = fig.add_subplot(gs[:2, 0])
    ax.set_title(f'{start_point["name"]} to {end_point["name"]}')
    LON, LAT = np.meshgrid(lon, lat)
    U, V = U[18][2], V[18][2]
    WIND = (U ** 2 + V ** 2) ** 0.5
    ax.pcolormesh(LON, LAT, WIND, cmap="jet")
    stride = 10
    ax.quiver(lon[::stride], lat[::stride], U[::stride, ::stride], V[::stride, ::stride], linewidth=0.1, color="gray")
    ax.scatter(end_point["coordinates"][1], end_point["coordinates"][0], marker='v', s=40, zorder=5)
    ax.scatter(start_point["coordinates"][1], start_point["coordinates"][0], marker='^', s=40, zorder=5)
    ax.set_xlabel('lon [deg]')
    ax.set_ylabel('lat [deg]')
    ax1, ax2, ax3, ax4, ax5 = ax, fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, 1]), fig.add_subplot(gs[2, 0]), fig.add_subplot(gs[2, 1])
    ax2.set_ylabel('reward')
    ax3.set_ylabel('flight time [s]')
    ax4.set_ylabel('altitude [ft]')
    ax5.set_ylabel('mass [kg]')

    ax4.set_facecolor('grey')
    return fig, ax1, ax2, ax3, ax4, ax5
