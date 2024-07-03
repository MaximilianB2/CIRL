import matplotlib.pyplot as plt
import pickle
import numpy as np

# Load network size analysis data
reps = 5
results_pid_network = []
for i in range(reps):
    with open(f'results_pid_network_rep_newobs_{i}.pkl', 'rb') as f:

        inter = pickle.load(f)
        r_list_values = []
        for d in range(len(inter)):
            r= [-1 * i for i in np.concatenate(inter[d]['r_list']).tolist()]
            r_list_values.append(r)
    results_pid_network.append(r_list_values)


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


# r_pid = [-1 * i for i in np.concatenate(results_pid_network[0]['r_list']).tolist()]

neurons = [16, 128]
x_values = np.linspace(0, 2355, 2355)[::6]
median_values_all = [median_net_i[i][::6] for i in range(len(median_net_i))]
min_values_all = [min_net_i[i][::6] for i in range(len(min_net_i))]
max_values_all = [max_net_i[i][::6] for i in range(len(max_net_i))]


results_rl_network = []
for i in range(reps):
    with open(f'results_rl_network_rep_newobs_{i}.pkl', 'rb') as f:

        inter = pickle.load(f)
        r_list_values = []
        for d in range(len(inter)):
            r= [-1 * i for i in np.concatenate(inter[d]['r_list']).tolist()]
            r_list_values.append(r)
    results_rl_network.append(r_list_values)


median_net_i_rl =[]
max_net_i_rl = []
min_net_i_rl = []
for i in range(len(inter)):
    net_i_rl = []
    for r_i in range(reps):   
        net_i_rl.append(results_rl_network[r_i][i])
    median_net_i_rl.append( np.median(np.array(net_i_rl),axis = 0))
    max_net_i_rl.append( np.max(np.array(net_i_rl),axis = 0))
    min_net_i_rl.append( np.min(np.array(net_i_rl),axis = 0))


# r_pid = [-1 * i for i in np.concatenate(results_pid_network[0]['r_list']).tolist()]
x_values = np.linspace(0, 2355,468)
import pandas as pd
neurons = [16, 128]
window_size = 10 # Define the size of the sliding window

# Convert lists to pandas Series for rolling window calculation
median_values_all = [pd.Series(median_net_i[i]).rolling(window=window_size).mean()[::window_size].tolist() for i in range(len(median_net_i))]
min_values_all = [pd.Series(min_net_i[i]).rolling(window=window_size).mean()[::window_size].tolist() for i in range(len(min_net_i))]
max_values_all = [pd.Series(max_net_i[i]).rolling(window=window_size).mean()[::window_size].tolist() for i in range(len(max_net_i))]

median_values_all_rl = [pd.Series(median_net_i_rl[i]).rolling(window=window_size).mean()[::window_size].tolist() for i in range(len(median_net_i_rl))]
min_values_all_rl = [pd.Series(min_net_i_rl[i]).rolling(window=window_size).mean()[::window_size].tolist() for i in range(len(min_net_i_rl))]
max_values_all_rl = [pd.Series(max_net_i_rl[i]).rolling(window=window_size).mean()[::window_size].tolist() for i in range(len(max_net_i_rl))]
# print(median_values_all_rl)
plt.figure(figsize=(8,8))
plt.rcParams["text.usetex"] = "True"
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.size"] = 16
col = ['tab:blue','tab:orange','tab:green','tab:red','tab:purple','tab:olive']
for i, n_i in enumerate(neurons):
    if i == 0 or i == 1:
        plt.plot(x_values, median_values_all[i], label = f'CIRL ({str(n_i)})',color= col[i])
        plt.fill_between(x_values,min_values_all[i],max_values_all[i],alpha =0.2,color = col[i],edgecolor = 'none')
        plt.plot(x_values, median_values_all_rl[i], label = f'RL ({str(n_i)})',color= col[i],linestyle = 'dashed')
        plt.fill_between(x_values,min_values_all_rl[i],max_values_all_rl[i],alpha =0.2,color = col[i],edgecolor = 'none')
    # plt.plot(np.linspace(0, 2355,2355), [-1 * i for i in np.concatenate(results_pid_network[i]['r_list']).tolist()], label = f'Pure-RL ({str(n_i)})')
plt.ylim(-100,0)
plt.xlim(0,2355)
plt.legend()
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.savefig('network_size_analysis.pdf')
plt.show()
