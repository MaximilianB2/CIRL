import matplotlib.pyplot as plt
import numpy as np
from CS1_Model import reactor_class
import torch
import torch.nn.functional as F

ns = 120
reps = 10


class Net(torch.nn.Module):
    def __init__(
        self,
        n_fc1,
        n_fc2,
        activation,
        n_layers,
        output_sz,
        deterministic,
        PID,
        **kwargs,
    ):
        super(Net, self).__init__()

        # Unpack the dictionary
        self.deterministic = deterministic
        self.args = kwargs
        self.dtype = torch.float
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device("cpu")
        self.pid = PID
        self.input_size = 8  # State size: Ca, T,Ca-,T- , Ca setpoint
        self.output_sz = output_sz  # Output size: Reactor Ks size
        self.n_layers = torch.nn.ModuleList()
        self.hs1 = n_fc1  # !! parameters
        self.hs2 = n_fc2  # !! parameter

        # defining layer
        self.hidden1 = torch.nn.Linear(self.input_size, self.hs1, bias=True)
        self.act = activation()
        self.hidden2 = torch.nn.Linear(self.hs1, self.hs2, bias=True)
        for i in range(0, n_layers):
            linear_layer = torch.nn.Linear(self.hs2, self.hs2)
            self.n_layers.append(linear_layer)
        self.output_mu = torch.nn.Linear(self.hs2, self.output_sz, bias=True)
        self.output_std = torch.nn.Linear(self.hs2, self.output_sz, bias=True)

    def forward(self, x):
        x = x.float()
        y = self.act(self.hidden1(x))
        y = self.act(self.hidden2(y))
        mu = self.output_mu(y)
        log_std = self.output_std(y)
        dist = torch.distributions.Normal(mu, log_std.exp() + 1e-6)
        out = dist.sample()
        out = torch.clamp(out, -1, 1)
        y = out
        if self.deterministic:
            if self.pid:
                y = F.tanh(mu)  # [-1,1]
            else:
                y = torch.clamp(mu, -1, 1)
            y = y.detach().numpy()

        return y


def rollout(env, best_policy, PID, PG=False, ES=False):
    Ca_eval_EA = np.zeros((ns, reps))
    T_eval_EA = np.zeros((ns, reps))
    Tc_eval_EA = np.zeros((ns, reps))
    F_eval_EA = np.zeros((ns, reps))
    V_eval_EA = np.zeros((ns, reps))
    ks_eval_EA = np.zeros((6, ns, reps))
    r_eval_EA = np.zeros((1, reps))

    for r_i in range(reps):
        s_norm, _ = env.reset()
        s = (
            s_norm * (env.observation_space.high - env.observation_space.low)
            + env.observation_space.low
        )
        Ca_eval_EA[0, r_i] = s[0]
        T_eval_EA[0, r_i] = s[1]
        V_eval_EA[0, r_i] = s[2]
        Tc_eval_EA[0, r_i] = 300.0
        F_eval_EA[0, r_i] = 100
        if PG:
            a_policy, _ = best_policy.predict(torch.tensor(s_norm), deterministic=True)
        elif ES:
            a_policy = best_policy(torch.tensor(s_norm))
        else:
            a_policy = best_policy
        x_norm = env.x_norm
        if PID:
            Ks_norm = ((a_policy + 1) / 2) * (x_norm[1] - x_norm[0]) + x_norm[0]
            ks_eval_EA[:, 0, r_i] = Ks_norm
        r_tot = 0
        for i in range(1, ns):
            if PG:
                a_policy, _ = best_policy.predict(
                    torch.tensor(s_norm), deterministic=True
                )
            elif ES:
                a_policy = best_policy(torch.tensor(s_norm))
            else:
                a_policy = best_policy
            if PID:
                Ks_norm = ((a_policy + 1) / 2) * (x_norm[1] - x_norm[0]) + x_norm[0]
                ks_eval_EA[:, i, r_i] = Ks_norm
            s_norm, r, done, info, _ = env.step(a_policy)

            r_tot += r
            s = (
                s_norm * (env.observation_space.high - env.observation_space.low)
                + env.observation_space.low
            )
            Ca_eval_EA[i, r_i] = s[0]
            T_eval_EA[i, r_i] = s[1]
            V_eval_EA[i, r_i] = s[2]
            Tc_eval_EA[i, r_i] = env.u_history[-1][0]
            F_eval_EA[i, r_i] = env.u_history[-1][1]
        r_eval_EA[:, r_i] = r_tot

    if PID and ES:
        print("CIRL (reward): ", np.round(-1 * np.mean(r_eval_EA), 2))
        return Ca_eval_EA, T_eval_EA, V_eval_EA, Tc_eval_EA, F_eval_EA, ks_eval_EA
    elif PID and not ES:
        print("static (reward): ", np.round(-1 * np.mean(r_eval_EA), 2))
        return Ca_eval_EA, T_eval_EA, V_eval_EA, Tc_eval_EA, F_eval_EA, ks_eval_EA
    else:
        print("RL (reward): ", np.round(-1 * np.mean(r_eval_EA), 2))
        return Ca_eval_EA, T_eval_EA, V_eval_EA, Tc_eval_EA, F_eval_EA


def plot_simulation_comp(
    Ca_dat_PG,
    T_dat_PG,
    Tc_dat_PG,
    Ca_dat_EA,
    T_dat_EA,
    Tc_dat_EA,
    ks_eval_EA,
    F_eval_EA,
    F_eval_PG,
    V_eval_EA,
    V_eval_PG,
    Ca_eval_PID_const,
    T_eval_PID_const,
    V_eval_PID_const,
    Tc_eval_PID_const,
    F_eval_PID_const,
    ks_eval_pid_const,
    SP,
    ns,
):
    plt.rcParams["text.usetex"] = "True"
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.size"] = 20
    t = np.linspace(0, 25, ns)
    fig, axs = plt.subplots(1, 2, figsize=(20, 10))
    labels = [
        "$Cb_{k_p}$",
        "$Cb_{k_i}$",
        "$Cb_{k_d}$",
        "$T_{k_p}$",
        "$T_{k_i}$",
        "$T_{k_d}$",
    ]
    col = ["tab:orange", "tab:red", "tab:blue", "tab:orange", "tab:red", "tab:blue"]
    col_fill = [
        "tab:orange",
        "tab:red",
        "tab:blue",
        "tab:orange",
        "tab:red",
        "tab:blue",
    ]
    Ca_des = SP[0]
    V_des = SP[1]
    axs[0].plot(t, np.median(Ca_dat_PG, axis=1), color="tab:red", lw=1, label="RL")
    axs[0].plot(t, np.median(Ca_dat_EA, axis=1), color="tab:blue", lw=1.5, label="CIRL")
    axs[0].plot(
        t,
        np.median(Ca_eval_PID_const, axis=1),
        color="tab:orange",
        lw=1.5,
        label="Static PID",
    )
    # axs[0].plot(t,np.median(Ca_eval_normRL_PG,axis=1), color = 'tab:green', lw=1.5, label = 'RL (PG)')
    # axs[0].plot(t,np.median(Ca_dat_GS,axis=1), color = 'tab:orange', lw=1.5, label = 'Offline GS')
    axs[0].fill_between(
        t,
        np.min(Ca_dat_PG, axis=1),
        np.max(Ca_dat_PG, axis=1),
        color="tab:red",
        alpha=0.1,
        edgecolor="none",
    )
    axs[0].fill_between(
        t,
        np.min(Ca_dat_EA, axis=1),
        np.max(Ca_dat_EA, axis=1),
        color="tab:blue",
        alpha=0.2,
        edgecolor="none",
    )

    # axs[0].fill_between(t, np.min(Ca_eval_normRL_PG,axis=1), np.max(Ca_eval_normRL_PG,axis=1),color = 'tab:green', alpha=0.2,edgecolor  = 'none')
    axs[0].fill_between(
        t,
        np.min(Ca_eval_PID_const, axis=1),
        np.max(Ca_eval_PID_const, axis=1),
        color="tab:orange",
        alpha=0.2,
        edgecolor="none",
    )
    axs[0].step(
        t, Ca_des, "--", lw=1.5, color="black", zorder=1, alpha=0.8, label="Setpoint"
    )

    # # Create a set of inset Axes: these should fill the bounding box allocated to them.
    # ax1_inset = inset_axes(axs[0], width="30%", height="30%", loc=2)
    # ax1_inset.plot(t, np.median(Ca_dat_PG,axis=1), color = 'tab:red', lw=1)
    # ax1_inset.plot(t,np.median(Ca_dat_EA,axis=1), color = 'tab:blue', lw=1.5)
    # ax1_inset.plot(t,np.median(Ca_eval_PID_const,axis=1), color = 'tab:orange', lw=1.5)
    # ax1_inset.step(t, Ca_des, '--', lw=1.5, color='black')
    # ax1_inset.set_xlim(16, 21)
    # ax1_inset.set_ylim(0.65, 0.8)
    # ax1_inset.set_xticklabels([])
    # ax1_inset.set_yticklabels([])
    # xyA_1 = (16, 0.8)
    # xyB = (0, 0)
    # coordsA = "data"
    # coordsB = "axes fraction"
    # con = ConnectionPatch(xyA=xyA_1, xyB=xyB, coordsA=coordsA, coordsB=coordsB, axesA=axs[0], axesB=ax1_inset)
    # axs[0].add_artist(con)

    # xyA = (21, max(np.median(Ca_dat_PG,axis=1)))
    # xyB = (1, 1)
    # con = ConnectionPatch(xyA=xyA, xyB=xyB, coordsA=coordsA, coordsB=coordsB, axesA=axs[0], axesB=ax1_inset)
    # axs[0].add_artist(con)
    axs[0].set_ylabel("Concentration of B, $C_B$ (mol/m$^3$)")
    axs[0].set_xlabel("Time (min)")

    # axs[0].legend(loc='lower right')
    axs[0].set_xlim(min(t), max(t))

    axs[0].grid(True, alpha=0.5)
    axs[0].set_axisbelow(True)

    axs[1].plot(t, np.median(V_eval_PG, axis=1), color="tab:red", lw=1, label="RL ")
    axs[1].plot(t, np.median(V_eval_EA, axis=1), color="tab:blue", lw=1, label="CIRL ")
    # axs[1].plot(t, np.median(V_eval_normRL_PG, axis=1), color = 'tab:green', lw=1, label = 'RL (PG)')
    axs[1].plot(
        t,
        np.median(V_eval_PID_const, axis=1),
        color="tab:orange",
        lw=1,
        label="Static PID",
    )
    axs[1].fill_between(
        t,
        np.min(V_eval_PG, axis=1),
        np.max(V_eval_PG, axis=1),
        color="tab:red",
        alpha=0.2,
        edgecolor="none",
    )
    axs[1].fill_between(
        t,
        np.min(V_eval_EA, axis=1),
        np.max(V_eval_EA, axis=1),
        color="tab:blue",
        alpha=0.2,
        edgecolor="none",
    )
    # axs[1].fill_between(t, np.min(V_eval_normRL_PG,axis=1), np.max(V_eval_normRL_PG,axis=1),color = 'tab:green', alpha=0.2,edgecolor  = 'none')
    axs[1].fill_between(
        t,
        np.min(V_eval_PID_const, axis=1),
        np.max(V_eval_PID_const, axis=1),
        color="tab:orange",
        alpha=0.2,
        edgecolor="none",
    )
    axs[1].step(t, V_des, "--", lw=1.5, color="black", label="Setpoint")
    axs[1].set_ylabel("Volume, $V$ (m$^3$)")
    axs[1].set_xlabel("Time (min)")
    axs[1].legend(
        bbox_to_anchor=(-0.1, 1.0),
        loc="lower center",
        ncol=4,
        frameon=False,
        columnspacing=0.5,
        fontsize=18,
        handletextpad=0.5,
    )
    axs[1].grid(True, alpha=0.5)
    axs[1].set_xlim(min(t), max(t))
    # axs[1].set_ylim(95,110)
    plt.subplots_adjust(wspace=0.3)
    plt.savefig("cs1_presentation.pdf")
    plt.show()


env = reactor_class(test=True, ns=120, PID_vel=True, normRL=True)
best_policy_rl = Net(
    n_fc1=128,
    n_fc2=128,
    activation=torch.nn.ReLU,
    n_layers=1,
    output_sz=2,
    PID=True,
    deterministic=True,
)
best_policy_rl.load_state_dict(torch.load("best_policy_rl_wnoise.pth"))
Ca_eval_RL, T_eval_RL, V_eval_RL, Tc_eval_RL, F_eval_RL = rollout(
    env, best_policy_rl, PID=False, ES=True
)


env = reactor_class(test=True, ns=120, PID_vel=True, normRL=False)
best_policy_pid = Net(
    n_fc1=128,
    n_fc2=128,
    activation=torch.nn.ReLU,
    n_layers=1,
    output_sz=6,
    PID=True,
    deterministic=True,
)
best_policy_pid.load_state_dict(torch.load("best_policy_pid_wnoise.pth"))
Ca_eval_pid, T_eval_pid, V_eval_pid, Tc_eval_pid, F_eval_pid, ks_eval_pid = rollout(
    env, best_policy_pid, PID=True, ES=True
)

env = reactor_class(test=True, ns=120, PID_vel=True, normRL=False)
best_policy_const_PID = np.load("constant_gains.npy")
(
    Ca_eval_PID_const,
    T_eval_PID_const,
    V_eval_PID_const,
    Tc_eval_PID_const,
    F_eval_PID_const,
    ks_eval_pid_const,
) = rollout(env, best_policy_const_PID, PID=True, PG=False)

# env = reactor_class(test = True, ns = 120, PID_vel= True, normRL=True)
# best_policy_PG_normRL = SAC.load('SAC_normRL_1604.zip')
# Ca_eval_normRL_PG,T_eval_normRL_PG,V_eval_normRL_PG,Tc_eval_normRL_PG,F_eval_normRL_PG = rollout(env,best_policy_PG_normRL,PID=False,PG=True)

SP = env.test_SP
plot_simulation_comp(
    Ca_eval_RL,
    T_eval_RL,
    Tc_eval_RL,
    Ca_eval_pid,
    T_eval_pid,
    Tc_eval_pid,
    ks_eval_pid,
    F_eval_pid,
    F_eval_RL,
    V_eval_pid,
    V_eval_RL,
    Ca_eval_PID_const,
    T_eval_PID_const,
    V_eval_PID_const,
    Tc_eval_PID_const,
    F_eval_PID_const,
    ks_eval_pid_const,
    SP,
    ns,
)
