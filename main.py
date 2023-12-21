import pandas as pd
from custom_environment import *
from utils import *
from DQN import *

#WIND SPEED/DIRECTION DATA
ds = nc.Dataset("Data/weather_data_17_04_2023.nc")
lon,lat,V,U,levels,times = ds.variables['longitude'][:],ds.variables['latitude'][:],ds.variables['v'][:],ds.variables['u'][:],ds.variables['level'][:],ds.variables['time'][:]

NEW_YORK = {'name':'New York','coordinates':(40,-74)}
LONDON = {'name':'London','coordinates':(51,0)}
start_point = LONDON
end_point = NEW_YORK

def create_lateral_input(state):
    input_matrix = np.zeros((state.grid_height, state.grid_width))
    if state.done_status is False:
        col_index = int((state.lon - state.min_lon) / state.grid_step)
        row_index = int((state.lat - state.min_lat) / state.grid_step)
        input_matrix[row_index, col_index] = 1
    return input_matrix.flatten()

def create_vertical_input(state):
    scaled_h = (state.h - min_alt)/(max_alt-min_alt)
    scaled_m = max(state.m/start_m,0)
    return np.array([scaled_m,scaled_h])

def train_agent(train_params, plot=False):
    N_ep, gamma, N_lateral_actions, N_lateral_hidden, N_vertical_actions, learning_rate, dx, step = [train_params[i] for i in train_params]

    initial_heading, lat_grid, lon_grid, grid_width, grid_height = get_grid_world(start_point, end_point, dx, step)

    initial_dict = {'start_coords': start_point['coordinates'], 'end_coords': end_point['coordinates'],
                    'V_TAS': V_TAS, 'initial_heading': initial_heading, 'initial_alt': start_alt, 'initial_m': start_m,
                    'min_m': min_m, 'min_alt': min_alt, 'max_alt': max_alt}

    grid_dict = {'dx': dx, 'step':step,
                 'max_lon': max(lon_grid), 'max_lat': max(lat_grid),
                 'min_lon': min(lon_grid), 'min_lat': min(lat_grid),
                 'grid_height':grid_height, 'grid_width':grid_width}

    env = Environment(initial_dict, grid_dict)

    # LATERAL ACTION SPACE
    lateral_action_space = np.linspace(initial_heading - 90, initial_heading + 90, N_lateral_actions) % 360
    lateral_qnet = lateral_QNetwork(n_inputs=int(grid_height * grid_width), n_hidden= N_lateral_hidden,
                                    n_outputs=N_lateral_actions, learning_rate=learning_rate)

    # VERTICAL ACTION SPACE
    vertical_action_space = np.linspace(0,1000,N_vertical_actions)
    vertical_qnet = vertical_QNetwork(n_inputs=2, n_outputs=N_vertical_actions,
                                      learning_rate=learning_rate)

    print(lateral_qnet)
    print(vertical_qnet)
    if plot:
        plt.ion()
        fig, ax1, ax2, ax3, ax4, ax5 = get_figure(start_point, end_point, lat, lon, U, V)

    solution_df = pd.DataFrame([],columns = ['rewards','epsilon','flight_time','fuel used'])
    epsilon = 1.

    for i in range(N_ep):
        s, ep_reward, ep_loss, j = env.reset(), 0, 0, 0
        flight_profile = pd.DataFrame([],columns = ['flight_time','lat','lon','altitude','lift-to-drag','ground_speed'])
        while True:
            # 1. do foward pass of current state to compute Q-values for all actions
            lateral_qnet.optimizer.zero_grad()
            vertical_qnet.optimizer.zero_grad()

            lateral_Q = lateral_qnet(torch.from_numpy(create_lateral_input(s)).float())
            vertical_Q = vertical_qnet(torch.from_numpy(create_vertical_input(s)).float())

            flight_profile.loc[j,:] = [s.flight_time, s.lat, s.lon, s.h, s.lift_to_drag, s.ground_speed]

            # 2. select action with epsilon-greedy strategy
            if np.random.rand() > epsilon:
                lateral_a = lateral_Q.argmax()
                vertical_a = vertical_Q.argmax()
            else:
                lateral_a = np.random.randint(0,len(lateral_action_space))
                vertical_a = np.random.randint(0,len(vertical_action_space))

            s1, lat_r, vert_r, done, terminal = env.step(lateral_action = lateral_action_space[lateral_a],
                                                         vertical_action = vertical_action_space[vertical_a])

            # 3. do forward pass for the next state
            with torch.no_grad():
                lateral_Q1 = lateral_qnet(torch.from_numpy(create_lateral_input(s1)).float())
                vertical_Q1 = vertical_qnet(torch.from_numpy(create_vertical_input(s1)).float())

            # 4. set Q-target
            lateral_q_target = lateral_Q.clone()
            lateral_q_target[lateral_a] = lat_r + gamma * lateral_Q1.max().item() * (not done) * (not terminal)

            vertical_q_target = vertical_Q.clone()
            vertical_q_target[vertical_a] = vert_r + gamma * vertical_Q1.max().item()

            # 5. update network weights
            lateral_loss = lateral_qnet.loss(lateral_Q, lateral_q_target)
            lateral_loss.backward()
            lateral_qnet.optimizer.step()

            vertical_loss = vertical_qnet.loss(vertical_Q, vertical_q_target)
            vertical_loss.backward()
            vertical_qnet.optimizer.step()

            # 6. bookkeeping
            s = s1
            ep_reward += lat_r
            ep_loss += lateral_loss.item()
            ep_reward += vert_r
            ep_loss += vertical_loss.item()
            j += 1

            if done:
                flight_time = np.nan
                fuel_used = np.nan
                break

            if terminal:
                flight_time = s.flight_time
                fuel_used = s.start_m - s.m
                break

        solution_df.loc[i,['reward','flight_time','fuel_used']] = ep_reward, flight_time, fuel_used
        # bookkeeping
        epsilon *= N_ep/(i/(N_ep/20)+N_ep) # decrease epsilon
        solution_df.loc[i,'epsilon'] = epsilon

        if (i+1) % 10 == 0:
            if i+1 == N_ep and plot:
                ax1.plot(flight_profile['lon'], flight_profile['lat'], c='white', linewidth = 4, zorder = 3)
                ax4.plot(flight_profile['flight_time'], flight_profile['altitude'], c='white',linewidth = 4, zorder = 3)

            try:
                print(f'epsiode {i + 1}: fuel_use = {solution_df["fuel_used"].dropna().rolling(20).mean().iloc[-1]} kg')
            except:
                print(f'episode {i + 1}: NaN')

            if plot:
                ax1.plot(flight_profile['lon'], flight_profile['lat'], c='k', alpha=max(1 - epsilon, 0.2))
                ax4.plot(flight_profile['flight_time'], flight_profile['altitude'], c='k',
                         alpha=max(1 - epsilon, 0.2))

                ax2.scatter(solution_df.index.values, solution_df['reward'], c='tab:blue', s=10, alpha=0.7)
                solution_df['reward'].rolling(50).mean().plot(ax=ax2, c='tab:orange')

                ax3.scatter(solution_df.index.values, solution_df['flight_time'], c='tab:blue', s=10, alpha=0.7)
                solution_df['flight_time'].dropna().rolling(20).mean().plot(ax=ax3, c='tab:orange')

                ax5.scatter(solution_df.index.values, solution_df['fuel_used'], c='tab:blue', s=10, alpha=0.7)
                solution_df['fuel_used'].dropna().rolling(20).mean().plot(ax=ax5, c='tab:orange')

                ax2.get_shared_x_axes().joined(ax2, ax3)
                ax3.get_shared_x_axes().joined(ax3, ax4)

                if i+1 != N_ep:
                    # re-drawing the figure
                    fig.canvas.draw()
                    # to flush the GUI events
                    fig.canvas.flush_events()

    if plot:
        # When the loop is done, the plot will stay open
        plt.ioff()  # Turn off interactive mode to regain control
        plt.show()  # This will keep the plot open until you manually close it

    converged_fuel_use = solution_df["fuel_used"].iloc[-1]
    converged_fuel_use_std = solution_df["fuel_used"].iloc[-50:].std()
    return converged_fuel_use, converged_fuel_use_std, flight_profile

V_TAS = 0.83*300    #True airspeed [m/s]
min_alt = 35000     #Min cruise altitude [ft]
max_alt = 40000     #Max cruise altitude [ft]
start_alt = 36000   #initial cruise altitude [ft]
start_m = 320000    #initial cruise mass [kg]
min_m = 230000      #minimal cruise mass if all fuel is used -> terminate [kg]

N_ep = 500
gamma = 0.98
N_lateral_actions = 30
N_lateral_hidden = 0 #N_lateral_actions
N_vertical_actions = 2
learning_rate = 0.1
dx = 100
step = 7
plot = True

train_params = {'N_ep':N_ep, 'gamma':gamma, 'N_lateral_actions':N_lateral_actions,'N_lateral_hidden':N_lateral_hidden,
                'N_vertical_actions':N_vertical_actions, 'learning_rate': learning_rate,
                'dx': dx, 'step': step}

file_name = " ".join([i+"_"+str(train_params[i]) for i in train_params])+".xlsx"
train_agent(train_params, plot=plot)
