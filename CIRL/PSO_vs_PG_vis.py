import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pickle

PPO_lc = []

for i in range(10):
    PPO_lc.append(pd.read_csv(f'PPO_PID_LC_rep_{i}.csv'))
combined_df = pd.concat(PPO_lc)
# print(combined_rewards)
# Calculate the median
median_rewards = combined_df.groupby('Episode')['Reward'].median()
max_rewards = combined_df.groupby('Episode')['Reward'].max()
min_rewards = combined_df.groupby('Episode')['Reward'].min()
# Load network size analysis data

results_pid_network = []
for i in range(3):
    with open(f'results_pid_network_rep_{i}.pkl', 'rb') as f:

        inter = pickle.load(f)
        r_list_values = []
        for d in range(len(inter)):
            r= [-1 * i for i in np.concatenate(inter[d]['r_list']).tolist()]
            r_list_values.append(r)
    results_pid_network.append(r_list_values)

reps = 3
median_net_i =[]
max_net_i = []
min_net_i = []
for i in range(len(inter)):
    net_i = []
    for r_i in range(reps):   
        net_i.append(results_pid_network[r_i][i])
    median_net_i.append( np.median(np.array(net_i),axis = 0))
    max_net_i.append( np.max(np.array(net_i),axis = 0))
    min_net_i.append( np.min(np.array(net_i),axis = 0))
# with open('results_rl_network.pkl', 'rb') as f:
#     results_rl_network = pickle.load(f)

# r_pid = [-1 * i for i in np.concatenate(results_pid_network[0]['r_list']).tolist()]

# neurons = [16, 32, 64, 128]
# window_size = 100  # Define the size of the sliding window

plt.figure()

# pid_rewards = [-1 * i for i in np.concatenate(results_pid_network[2]['r_list']).tolist()]
# rl_rewards = [-1 * i for i in np.concatenate(results_rl_network[2]['r_list']).tolist()]

# Calculate the rolling mean
# pid_rewards_rolling_mean = pd.Series(pid_rewards).rolling(window=window_size).mean()
# rl_rewards_rolling_mean = pd.Series(rl_rewards).rolling(window=window_size).mean()

# plt.plot(np.linspace(0, 2355,2355), median_net_i[3], label = f'CIRL (64)', color = 'tab:blue')
# plt.fill_between(np.linspace(0, 2355,2355),min_net_i[3],max_net_i[3], color = 'tab:blue',alpha =0.2)
x_values = np.linspace(0, 2355, 236)

window_size =10
median_values = [pd.Series(median_net_i[i]).rolling(window=window_size).mean()[::window_size].tolist() for i in range(len(median_net_i))]
min_values = [pd.Series(min_net_i[i]).rolling(window=window_size).mean()[::window_size].tolist() for i in range(len(min_net_i))]
max_values = [pd.Series(max_net_i[i]).rolling(window=window_size).mean()[::window_size].tolist() for i in range(len(max_net_i))]

median_rewards = pd.Series(median_rewards).rolling(window=window_size).mean()
min_rewards = pd.Series(min_rewards).rolling(window=window_size).mean()
max_rewards = pd.Series(max_rewards).rolling(window=window_size).mean()
# print(median_values_all_rl)

plt.plot(x_values, median_values[1], label = f'CIRL (16)', color = 'tab:blue')
plt.fill_between(x_values, min_values[1], max_values[1], color = 'tab:blue', alpha =0.2,edgecolor = 'none')
# plt.plot(np.linspace(0, 4680,4680), rl_rewards, label = f'Pure-RL (64)')

# Calculate the rolling mean for PPO_lc['Reward']
plt.plot(PPO_lc[0]['Episode']*6, median_rewards, label = 'PPO', color = 'tab:orange')
plt.fill_between(PPO_lc[0]['Episode']*6,min_rewards,max_rewards,alpha = 0.2, color  = 'tab:orange', edgecolor = 'none')
plt.ylim(-100,0)
plt.xlim(0,2355)
plt.legend()
plt.xlabel('Episodes')
plt.ylabel('Reward')
plt.savefig('ppo_vs_cirl.pdf')
plt.show()