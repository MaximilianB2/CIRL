import matplotlib.pyplot as plt
import pickle
import numpy as np

# Open and read reward files
with open("r_pid.pkl", "rb") as f:
    r_pid = pickle.load(f)

# Load r_rl list
with open("r_rl.pkl", "rb") as f:
    r_rl = pickle.load(f)

# Remove the first element since all start at 1e8 and multiply by -1 to show reward not cost

r_pid = np.median(np.array(r_pid), axis=2)
r_rl = np.median(np.array(r_rl), axis=2)

# print(np.median(np.median(r_pid,axis=2),axis=0).shape)

plt.rcdefaults()
plt.figure(figsize=(7, 5), layout="constrained")
plt.rcParams["text.usetex"] = "True"
plt.rcParams["font.family"] = "serif"
plt.plot(
    np.linspace(0, r_pid.shape[1], r_pid.shape[1]),
    np.median(r_pid * -1, axis=0),
    color="tab:blue",
    label="CIRL (Median)",
)
plt.fill_between(
    np.linspace(0, r_pid.shape[1], r_pid.shape[1]),
    np.min(r_pid * -1, axis=0),
    np.max(r_pid * -1, axis=0),
    color="tab:blue",
    edgecolor="none",
    alpha=0.3,
)
plt.plot(
    np.linspace(0, r_rl.shape[1], r_rl.shape[1]),
    np.median(r_rl * -1, axis=0),
    color="tab:red",
    label="RL (Median)",
)
plt.fill_between(
    np.linspace(0, r_rl.shape[1], r_rl.shape[1]),
    np.min(r_rl * -1, axis=0),
    np.max(r_rl * -1, axis=0),
    color="tab:red",
    edgecolor="none",
    alpha=0.3,
)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.xlabel("Iterations", fontsize=18)
plt.ylabel("Reward", fontsize=18)
plt.grid("True", alpha=0.4)
plt.xlim(0, 100)
plt.ylim(-100, 0)
plt.legend(loc="lower right", fontsize=14)
plt.savefig("Learning_curve_RLPID.pdf")
plt.show()
