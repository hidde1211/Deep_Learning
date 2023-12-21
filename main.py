import pandas as pd
from custom_environment import *
from utils import *
from DQN import *

#WIND SPEED/DIRECTION DATA
ds = nc.Dataset("../Data/weather_data_17_04_2023.nc")
lon,lat,V,U,levels,times = ds.variables['longitude'][:],ds.variables['latitude'][:],ds.variables['v'][:],ds.variables['u'][:],ds.variables['level'][:],ds.variables['time'][:]

NEW_YORK = {'name':'New York','coordinates':(40,-74)}
LONDON = {'name':'London','coordinates':(51,0)}

start_point = LONDON
end_point = NEW_YORK

dx = 100
V_TAS = 0.83*300
start_alt = 36000
start_m = 320000
step = 5

initial_heading, lat_grid, lon_grid, grid_width, grid_height = get_grid_world(start_point,end_point,dx, step)

initial_dict = {'start_coords':start_point['coordinates'],'end_coords':end_point['coordinates'],
                'V_TAS':V_TAS, 'initial_heading':initial_heading,'initial_alt':start_alt,'initial_m':start_m}
grid_dict = {'dx':dx,'max_lon':max(lon_grid),'max_lat':max(lat_grid),'min_lon':min(lon_grid),'min_lat':min(lat_grid)}

#inputs = position w.r.t. desitnation (distance and angle), heading angle, altitude, climb angle and mass
def create_NN_input(state):
    input_matrix = np.zeros((grid_height, grid_width))
    if state.done_status is False:
        col_index = int((state.lon - min(lon_grid)) / step)
        row_index = int((state.lat - min(lat_grid)) / step)
        input_matrix[row_index, col_index] = 1
    return input_matrix.flatten() #np.append(input_matrix.flatten(), [state.h - state.start_alt])

n_inputs = grid_height*grid_width

#actions = heading and climb angle
heading_space = np.linspace(initial_heading-90,initial_heading+90,31)%360
climb_space = np.array([0])
HEADING, CLIMB = np.meshgrid(heading_space,climb_space)
action_space = np.column_stack((HEADING.flatten(), CLIMB.flatten()))
n_actions = len(action_space)
env = Environment(n_inputs, n_actions, initial_dict, grid_dict)

learning_rate = 0.1
qnet = QNetwork(n_inputs, n_actions, learning_rate)

N_ep = 200
val_freq = max(int(N_ep/10),1)
epsilon = 1.
gamma = 0.95
plot = True

if plot:
    plt.ion()
    fig, ax1, ax2, ax3, ax4, ax5 = get_figure(start_point, end_point, lat, lon, U, V)

solution_df = pd.DataFrame([],columns = ['rewards','epsilon','flight_time','fuel used'])
for i in range(N_ep):
    s, ep_reward, ep_loss = env.reset(), 0, 0
    flight_profile = []
    while True:
        # 1. do foward pass of current state to compute Q-values for all actions
        qnet.optimizer.zero_grad()
        Q = qnet(torch.from_numpy(create_NN_input(s)).float())
        flight_profile.append([s.lat, s.lon, s.h, s.flight_time])

        # 2. select action with epsilon-greedy strategy
        a = Q.argmax() if np.random.rand() > epsilon else env.action_space.sample()
        s1, r, done, terminal = env.step(action = action_space[a])

        # 3. do forward pass for the next state
        with torch.no_grad():
            Q1 = qnet(torch.from_numpy(create_NN_input(s1)).float())

        # 4. set Q-target
        q_target = Q.clone()
        q_target[a] = r + gamma * Q1.max().item() * (not done) * (not terminal)

        # 5. update network weights
        loss = qnet.loss(Q, q_target)
        loss.backward()
        qnet.optimizer.step()

        # 6. bookkeeping
        s = s1
        ep_reward += r
        ep_loss += loss.item()

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

    if (i+1) % 50 == 0:
        flight_profile = np.array(flight_profile)
        ax1.plot(flight_profile[:, 1], flight_profile[:, 0], c='k', alpha=max(1 - epsilon, 0.2))
        ax4.plot(flight_profile[:, -1], flight_profile[:, 2], c='k', alpha=max(1 - epsilon, 0.2))

        if i+1 == N_ep:
            ax1.plot(flight_profile[:, 1], flight_profile[:, 0], c='white', linewidth = 4, zorder = 3)
            ax4.plot(flight_profile[:, -1], flight_profile[:, 2], c='white',linewidth = 4, zorder = 3)

    if (i+1) % 10 == 0:
        print(f'epsiode {i + 1}: r = {ep_reward} , fuel_use = {fuel_used} kg')
        if plot:
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

