import torch
import torch.nn.functional as F
from cstr_model import reactor_class
from torch_pso import ParticleSwarmOptimizer
import copy
import numpy as np
import matplotlib.pyplot as plt
from cirl_policy import Net as cirl_net

ns = 120
reps = 10


def rollout(env, best_policy, PID, PG=False):
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
        else:
            a_policy = best_policy(torch.tensor(s_norm))
        x_norm = env.x_norm
        if PID:
            Ks_norm = ((a_policy.detach().numpy() + 1) / 2) * (
                x_norm[1] - x_norm[0]
            ) + x_norm[0]
            ks_eval_EA[:, 0, r_i] = Ks_norm
        r_tot = 0
        for i in range(1, ns):
            if PG:
                a_policy, _ = best_policy.predict(
                    torch.tensor(s_norm), deterministic=True
                )
            else:
                a_policy = best_policy(torch.tensor(s_norm))
            if PID:
                Ks_norm = ((a_policy.detach().numpy() + 1) / 2) * (
                    x_norm[1] - x_norm[0]
                ) + x_norm[0]
                ks_eval_EA[:, i, r_i] = Ks_norm
            try:
                s_norm, r, done, info, _ = env.step(a_policy)
            except Exception:
                s_norm, r, done, info, _ = env.step(a_policy.detach().numpy())
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

    if PID:
        print("CIRL (reward): ", np.round(-1 * np.mean(r_eval_EA), 2))
        return Ca_eval_EA, T_eval_EA, V_eval_EA, Tc_eval_EA, F_eval_EA, ks_eval_EA
    else:
        print("RL (reward): ", np.round(-1 * np.mean(r_eval_EA), 2))
        return Ca_eval_EA, T_eval_EA, V_eval_EA, Tc_eval_EA, F_eval_EA


def plot_simulation_comp(
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
    SP,
    ns,
):
    plt.rcParams["text.usetex"] = "True"
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.size"] = 20
    t = np.linspace(0, 25, ns)
    fig, axs = plt.subplots(1, 2, figsize=(20, 10))
    Ca_des = SP[0]

    axs[0].plot(
        t, np.median(Ca_eval_RL, axis=1), color="tab:red", lw=1, label="RL (non-obs)"
    )
    axs[0].plot(
        t,
        np.median(Ca_eval_pid, axis=1),
        color="tab:blue",
        lw=1.5,
        label="CIRL (non-obs)",
    )

    axs[0].fill_between(
        t,
        np.min(Ca_eval_RL, axis=1),
        np.max(Ca_eval_RL, axis=1),
        color="tab:red",
        alpha=0.2,
    )
    axs[0].fill_between(
        t,
        np.min(Ca_eval_pid, axis=1),
        np.max(Ca_eval_pid, axis=1),
        color="tab:blue",
        alpha=0.2,
    )

    axs[0].step(t, Ca_des, "--", lw=1.5, color="black")
    axs[0].set_ylabel("Concentration of B, $C_B$ (mol/m$^3$)")
    axs[0].set_xlabel("Time (min)")

    axs[0].set_xlim(min(t), max(t))
    axs[0].grid(True, alpha=0.5)
    axs[0].set_axisbelow(True)

    axs[1].step(
        t,
        np.median(Tc_eval_RL, axis=1),
        color="tab:red",
        lw=1,
        label="RL (non-obs)",
        linestyle="dashed",
    )
    axs[1].step(
        t,
        np.median(Tc_eval_pid, axis=1),
        color="tab:blue",
        lw=1,
        label="CIRL (non-obs)",
        linestyle="dashed",
    )

    axs[1].fill_between(
        t,
        np.min(Tc_eval_RL, axis=1),
        np.max(Tc_eval_RL, axis=1),
        color="tab:red",
        alpha=0.2,
        edgecolor="none",
    )
    axs[1].fill_between(
        t,
        np.min(Tc_eval_pid, axis=1),
        np.max(Tc_eval_pid, axis=1),
        color="tab:blue",
        alpha=0.2,
        edgecolor="none",
    )

    axs[1].set_ylabel("Cooling Temperature, $T_C$ (K)")
    axs[1].set_xlabel("Time (min)")

    axs[1].grid(True, alpha=0.5)
    axs[1].set_xlim(min(t), max(t))
    axs[1].legend(
        bbox_to_anchor=(-0.1, 1.0),
        loc="lower center",
        ncol=5,
        frameon=False,
        columnspacing=0.5,
        fontsize=18,
        handletextpad=0.5,
    )
    # axs[1].set_ylim(95,110)

    plt.subplots_adjust(wspace=0.3)
    plt.savefig("disturbance_presentation.pdf")
    plt.show()
    plt.show()

if __name__ == '__main__':
    env = reactor_class(test=True, ns=120, normRL=True, dist=True)
    best_policy_rl = cirl_net(
        n_fc1=128,
        n_fc2=128,
        activation=torch.nn.ReLU,
        n_layers=1,
        output_sz=2,
        input_sz=15,
        deterministic=True,
        PID=True,
    )
    best_policy_rl.load_state_dict(torch.load("./data/best_policy_rl_dist.pth"))
    Ca_eval_RL, T_eval_RL, V_eval_RL, Tc_eval_RL, F_eval_RL = rollout(
        env, best_policy_rl, PID=False
    )


    env = reactor_class(test=True, ns=120, normRL=False, dist=True)
    best_policy_pid = cirl_net(
        n_fc1=16,
        n_fc2=16,
        activation=torch.nn.ReLU,
        n_layers=1,
        output_sz=6,
        input_sz=15,
        deterministic=True,
        PID=True,
    )
    best_policy_pid.load_state_dict(torch.load("./data/best_policy_pid_dist.pth"))
    Ca_eval_pid, T_eval_pid, V_eval_pid, Tc_eval_pid, F_eval_pid, ks_eval_pid = rollout(
        env, best_policy_pid, PID=True
    )

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
        SP,
        ns,
    )
