import matplotlib.pyplot as plt
import pickle
import numpy as np

#  Open and read reward files
with open("r_pid_dist_obs.pkl", "rb") as f:
    r_pid_dist_obs = pickle.load(f)

with open("r_rl_dist_obs.pkl", "rb") as f:
    r_rl_dist_obs = pickle.load(f)

with open("r_pid_dist_nonobs.pkl", "rb") as f:
    r_pid_dist_nonobs = pickle.load(f)


with open("r_rl_dist_nonobs.pkl", "rb") as f:
    r_rl_dist_nonobs = pickle.load(f)

#  Remove the first element since all start at 1e8 and multiply by -1 to show reward not cost
r_pid_dist_obs = np.array(r_pid_dist_obs)
r_rl_dist_obs = np.array(r_rl_dist_obs)
r_pid_dist_nonobs = np.array(r_pid_dist_nonobs)
r_rl_dist_nonobs = np.array(r_rl_dist_nonobs)


plt.rcdefaults()
plt.figure(figsize=(7, 5), layout="constrained")
plt.rcParams["text.usetex"] = "True"
plt.rcParams["font.family"] = "serif"
plt.plot(
    np.linspace(0, r_pid_dist_obs.shape[1], r_pid_dist_obs.shape[1]),
    np.median(r_pid_dist_obs[0] * -1, axis=1),
    color="tab:blue",
    label="CA-RL (Obs, Median)",
)
# plt.fill_between(np.linspace(0,r_pid_dist_obs.shape[1],r_pid_dist_obs.shape[1]), np.min(r_pid_dist_obs[0]*-1,axis =1), np.max(r_pid_dist_obs[0]*-1,axis =1),color = 'tab:blue', edgecolor = 'none',alpha=0.3)

# plt.fill_between(np.linspace(0,r_rl_dist_obs.shape[1],r_rl_dist_obs.shape[1]), np.min(r_rl_dist_obs[0]*-1,axis =1), np.max(r_rl_dist_obs[0]*-1,axis =1),color = 'tab:red', edgecolor = 'none',alpha=0.3)

plt.plot(
    np.linspace(0, r_pid_dist_nonobs.shape[1], r_pid_dist_nonobs.shape[1]),
    np.median(r_pid_dist_nonobs[0] * -1, axis=1),
    color="tab:cyan",
    label="CA-RL (Non-obs, Median)",
)
# plt.fill_between(np.linspace(0,r_pid_dist_nonobs.shape[1],r_pid_dist_nonobs.shape[1]), np.min(r_pid_dist_nonobs[0]*-1,axis =1), np.max(r_pid_dist_nonobs[0]*-1,axis =1),color = 'tab:cyan', edgecolor = 'none',alpha=0.3)

plt.plot(
    np.linspace(0, r_rl_dist_obs.shape[1], r_rl_dist_obs.shape[1]),
    np.median(r_rl_dist_obs[0] * -1, axis=1),
    color="tab:red",
    label="RL (Obs, Median)",
)
plt.plot(
    np.linspace(0, r_rl_dist_nonobs.shape[1], r_rl_dist_nonobs.shape[1]),
    np.median(r_rl_dist_nonobs[0] * -1, axis=1),
    color="tab:orange",
    label="RL (Non-obs, Median)",
)
# plt.fill_between(np.linspace(0,r_rl_dist_nonobs.shape[1],r_rl_dist_nonobs.shape[1]), np.min(r_rl_dist_nonobs[0]*-1,axis =1), np.max(r_rl_dist_nonobs[0]*-1,axis =1),color = 'tab:red', edgecolor = 'none',alpha=0.3)

plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.xlabel("Iterations", fontsize=18)
plt.ylabel("Reward", fontsize=18)
plt.grid("True", alpha=0.4)
plt.xlim(0, 100)
plt.ylim(-200, 0)
plt.legend(loc="lower right", fontsize=14)
plt.savefig("Learning_curve_RLPID_dist.pdf")
plt.show()
