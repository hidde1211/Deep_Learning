import numpy as np
import gymnasium as gym
from scipy.interpolate import RegularGridInterpolator
import geopy
import geopy.distance
import matplotlib.pyplot as plt
import math
import netCDF4 as nc
from ambiance import Atmosphere

ds = nc.Dataset("../Data/weather_data_17_04_2023.nc")
lon,lat,V,U,levels,times = ds.variables['longitude'][:],ds.variables['latitude'][:],ds.variables['v'][:],ds.variables['u'][:],ds.variables['level'][:],ds.variables['time'][:]
h_levels = ((1/(-0.000157688))*np.log(levels/(0.223356*1013.25)) + 11000)*3.2808399


def get_bearing(lat1,lon1,lat2,lon2):
    dLon = lon2 - lon1;
    y = math.sin(dLon) * math.cos(lat2);
    x = math.cos(lat1)*math.sin(lat2) - math.sin(lat1)*math.cos(lat2)*math.cos(dLon);
    brng = np.rad2deg(math.atan2(y, x));
    if brng < 0: brng+= 360
    return brng - 90

def get_ground_speed(self):
    cartesean_heading = -(self.phi - 90)
    x_ac, y_ac = np.cos((cartesean_heading * np.pi / 180)), np.sin(cartesean_heading * np.pi / 180)
    U_interp = RegularGridInterpolator((h_levels, lat, lon), U[18])
    V_interp = RegularGridInterpolator((h_levels, lat, lon), V[18])
    u, v = U_interp([self.h, self.lat, self.lon]), V_interp([self.h, self.lat,self.lon])
    headwind = x_ac * u + y_ac * v
    V_GS = headwind + self.V_TAS
    return V_GS[0]

def get_flight_performance(self):
    # GET ATMOSPHERIC STATE
    h_m = self.h * 0.3048
    rho = Atmosphere(h_m).density[0]
    CD0, k = 0.013, 0.047  # drag polar
    S = 436.8  # surface area [m2]
    TSFC = 16  # thrust specific fuel consumption druing cruise [g/kNs]

    CL = self.m * 9.81 / (0.5 * rho * self.V_TAS * self.V_TAS * S)
    CD = CD0 + k * CL ** 2

    Tr = np.maximum(CD * 0.5 * rho * self.V_TAS * self.V_TAS * S + self.m * 9.81 * np.deg2rad(self.gamma), 0.)
    mf_dot = TSFC * Tr / (1 * 10 ** 6)
    return CL/CD, mf_dot

class Environment(gym.Env):
    def __init__(self, n_inputs, n_actions, initial_dict, grid_dict):
        super(Environment, self).__init__()

        # Define observation space (state)
        self.observation_space = gym.spaces.Discrete(n_inputs)
        self.observation_space.n = n_inputs
        self.max_lat = grid_dict['max_lat']
        self.min_lat = grid_dict['min_lat']
        self.min_lon = grid_dict['min_lon']
        self.max_lon = grid_dict['max_lon']
        self.start_lat = initial_dict['start_coords'][0]
        self.start_lon = initial_dict['start_coords'][1]
        self.end_lat = initial_dict['end_coords'][0]
        self.end_lon = initial_dict['end_coords'][1]
        self.start_phi = initial_dict['initial_heading']
        self.start_alt = initial_dict['initial_alt']
        self.start_m = initial_dict['initial_m']

        # Define action space (discrete changes in heading angle)
        self.n_inputs = n_inputs
        self.action_space = gym.spaces.Discrete(n_actions)
        self.action_space.n = n_actions
        self.dx = grid_dict['dx']
        self.V_TAS = initial_dict['V_TAS']

        # State variables
        self.lat = self.start_lat
        self.lon = self.start_lon

        #heading and flight climb angle
        self.phi = self.start_phi
        self.gamma = 0

        #altitude, mass and flight time
        self.h = self.start_alt
        self.m = self.start_m
        self.flight_time = 0

    def reset(self):
        # Reset the environment to the initial state
        self.lat = self.start_lat
        self.lon = self.start_lon
        self.phi = self.start_phi
        self.h = self.start_alt
        self.m = self.start_m
        self.flight_time = 0
        self.done_status = False
        return self

    def step(self, action):
        # Update the trajectory angles
        self.phi, self.delta_h = action

        self.gamma = self.delta_h*.0003048/self.dx

        #get current ground speed for trajectory
        self.ground_speed = get_ground_speed(self)

        # Update the position
        new_point = geopy.distance.geodesic(kilometers=self.dx).destination(geopy.Point(self.lat, self.lon), self.phi)
        self.lat = new_point.latitude
        self.lon = new_point.longitude
        self.h += self.delta_h
        if self.h>max(h_levels):
            self.h = max(h_levels)
        if self.h<min(h_levels):
            self.h = min(h_levels)

        # get current ground speed for trajectory
        self.av_ground_speed = (get_ground_speed(self) + self.ground_speed)/2

        #get distance to destination
        self.dist_to_end = geopy.distance.geodesic((self.lat,self.lon),(self.end_lat,self.end_lon)).kilometers

        # Get flight performance parameters
        self.lift_to_drag, self.mfdot = get_flight_performance(self)

        #update mass and flight time
        self.m += -(self.mfdot/self.ground_speed)*self.dx*1000
        self.flight_time += self.dx*1000/self.av_ground_speed

        # Check if the episode is done if the agent reaches the destination, reward positively
        terminal = self._is_terminal()
        self.terminal_status = terminal

        # Check if the agent is out of bounds. If so, end episode and reward negatively
        done = self._is_done()
        self.done_status = done

        # Get reward based on if state is done or in progress
        reward = self._get_reward()
        return self, reward, done, terminal

    def _is_done(self):
        if self.lat > self.max_lat or self.lat < self.min_lat or self.lon > self.max_lon or self.lon < self.min_lon:
            return True
        else:
            return False

    def _is_terminal(self):
        if self.dist_to_end < 200:
            return True
        else:
            return False

    def _get_reward(self):
        if self._is_done():
            return -1000

        elif self.terminal_status:
            return min(int(1000*np.exp(-0.0001*(self.start_m - self.m -75000))),3000)

        else:
            return -1