import matplotlib.pyplot as plt
import pickle
import numpy as np


# Load the data from the pickle file
with open('r_pid.pkl', 'rb') as f:
    r_pid = pickle.load(f)
reps = 10
r_pid_all = np.zeros((2355,reps))

for r_i in range(reps):
  r_pid_all[:,r_i] = np.concatenate(r_pid[r_i])


print(r_pid_all.shape)
plt.figure()
plt.plot(np.linspace(0,2355,2355),np.median(r_pid_all,axis=1)*-1,color = 'tab:blue')
plt.fill_between(np.linspace(0,2355,2355),np.min(r_pid_all,axis=1)*-1,np.max(r_pid_all,axis=1)*-1, alpha =0.2, color = 'tab:blue')
plt.ylim(-50,0)
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
