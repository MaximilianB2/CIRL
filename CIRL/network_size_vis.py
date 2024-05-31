import matplotlib.pyplot as plt
import pickle
import numpy as np

# Load network size analysis data
with open('results_pid_network.pkl', 'rb') as f:
    results_pid_network = pickle.load(f)

with open('results_rl_network.pkl', 'rb') as f:
    results_rl_network = pickle.load(f)
r_pid = [-1 * i for i in np.concatenate(results_pid_network[0]['r_list']).tolist()]

neurons = [16, 32, 64, 128]
plt.figure()
for i, n_i in enumerate(neurons):
    plt.plot(np.linspace(0, 4680,4680), [-1 * i for i in np.concatenate(results_pid_network[i]['r_list']).tolist()], label = f'CIRL ({str(n_i)})')
    plt.plot(np.linspace(0, 4680,4680), [-1 * i for i in np.concatenate(results_pid_network[i]['r_list']).tolist()], label = f'Pure-RL ({str(n_i)})')
plt.ylim(-20,0)
plt.xlim
plt.legend()
plt.show()