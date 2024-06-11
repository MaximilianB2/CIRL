import matplotlib.pyplot as plt
import pickle
import numpy as np

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


# r_pid = [-1 * i for i in np.concatenate(results_pid_network[0]['r_list']).tolist()]

neurons = [8,16, 32, 64, 128,256]
x_values = np.linspace(0, 2355, 2355)[::6]
median_values_all = [median_net_i[i][::6] for i in range(len(median_net_i))]
min_values_all = [min_net_i[i][::6] for i in range(len(min_net_i))]
max_values_all = [max_net_i[i][::6] for i in range(len(max_net_i))]


results_rl_network = []
for i in range(3):
    with open(f'results_rl_network_rep_{i}.pkl', 'rb') as f:

        inter = pickle.load(f)
        r_list_values = []
        for d in range(len(inter)):
            r= [-1 * i for i in np.concatenate(inter[d]['r_list']).tolist()]
            r_list_values.append(r)
    results_rl_network.append(r_list_values)

reps = 3
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

neurons = [8,16, 32, 64, 128,256]
x_values = np.linspace(0, 2355,236)
import pandas as pd

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
    if i == 1:
        plt.plot(x_values, median_values_all[i], label = f'CIRL ({str(n_i)})',color= 'tab:blue')
        plt.fill_between(x_values,min_values_all[i],max_values_all[i],alpha =0.2,color = 'tab:blue', edgecolor = 'none')
    if i == 4:
        plt.plot(x_values, median_values_all_rl[i], label = f'RL ({str(n_i)})',color= 'tab:red')
        plt.fill_between(x_values,min_values_all_rl[i],max_values_all_rl[i],alpha =0.2,color = 'tab:red', edgecolor = 'none')
    # plt.plot(np.linspace(0, 2355,2355), [-1 * i for i in np.concatenate(results_pid_network[i]['r_list']).tolist()], label = f'Pure-RL ({str(n_i)})')
plt.ylim(-100,0)
plt.xlim(0,2355)
plt.legend()
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.savefig('lc_sp_0306.pdf')
plt.show()


# print(np.median(np.median(r_pid,axis=2),axis=0).shape)

# plt.rcdefaults()
# plt.figure(figsize=(7, 5), layout="constrained")
# plt.rcParams["text.usetex"] = "True"
# plt.rcParams["font.family"] = "serif"
# plt.plot(
#     np.linspace(0, r_pid.shape[1], r_pid.shape[1]),
#     np.median(r_pid * -1, axis=0),
#     color="tab:blue",
#     label="CIRL (Median)",
# )
# plt.fill_between(
#     np.linspace(0, r_pid.shape[1], r_pid.shape[1]),
#     np.min(r_pid * -1, axis=0),
#     np.max(r_pid * -1, axis=0),
#     color="tab:blue",
#     edgecolor="none",
#     alpha=0.3,
# )
# plt.plot(
#     np.linspace(0, r_rl.shape[1], r_rl.shape[1]),
#     np.median(r_rl * -1, axis=0),
#     color="tab:red",
#     label="RL (Median)",
# )
# plt.fill_between(
#     np.linspace(0, r_rl.shape[1], r_rl.shape[1]),
#     np.min(r_rl * -1, axis=0),
#     np.max(r_rl * -1, axis=0),
#     color="tab:red",
#     edgecolor="none",
#     alpha=0.3,
# )
# plt.xticks(fontsize=18)
# plt.yticks(fontsize=18)
# plt.xlabel("Iterations", fontsize=18)
# plt.ylabel("Reward", fontsize=18)
# plt.grid("True", alpha=0.4)
# plt.xlim(0, 100)
# plt.ylim(-100, 0)
# plt.legend(loc="lower right", fontsize=14)
# plt.savefig("Learning_curve_RLPID.pdf")
# plt.show()
