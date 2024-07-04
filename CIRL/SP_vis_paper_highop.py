import matplotlib.pyplot as plt
import numpy as np
from cstr_model import reactor_class
import torch
import torch.nn.functional as F
from cirl_policy import Net as cirl_net

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
            try:
                Ks_norm = ((a_policy + 1) / 2) * (x_norm[1] - x_norm[0]) + x_norm[0]
            except Exception:
                Ks_norm = ((a_policy.detach().numpy()+ 1) / 2) * (x_norm[1] - x_norm[0]) + x_norm[0]
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
                try:
                    Ks_norm = ((a_policy + 1) / 2) * (x_norm[1] - x_norm[0]) + x_norm[0]
                except Exception:
                    Ks_norm = ((a_policy.detach().numpy()+ 1) / 2) * (x_norm[1] - x_norm[0]) + x_norm[0]
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
        print("CIRL(reward): ", np.round(-1 * np.mean(r_eval_EA), 2))
        return Ca_eval_EA, T_eval_EA, V_eval_EA, Tc_eval_EA, F_eval_EA, ks_eval_EA
    elif PID and not ES:
        print("static (reward): ", np.round(-1 * np.mean(r_eval_EA), 2))
        return Ca_eval_EA, T_eval_EA, V_eval_EA, Tc_eval_EA, F_eval_EA, ks_eval_EA
    else:
        print("RL (reward): ", np.round(-1 * np.mean(r_eval_EA), 2))
        return Ca_eval_EA, T_eval_EA, V_eval_EA, Tc_eval_EA, F_eval_EA


def plot_simulation_comp(
    Ca_dat_EA,
    T_dat_EA,
    Tc_dat_EA,
    ks_eval_EA,
    F_eval_EA,
    V_eval_EA,
    Ca_eval_PID_const,
    T_eval_PID_const,
    V_eval_PID_const,
    Tc_eval_PID_const,
    F_eval_PID_const,
    ks_eval_pid_const,
    Ca_eval_pid_lowop,
    T_eval_pid_lowop,
    V_eval_pid_lowop,
    Tc_eval_pid_lowop,
    F_eval_pid_lowop,
    ks_eval_pid_lowop,
    SP,
    ns,
):
    plt.rcParams["text.usetex"] = "True"
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.size"] = 16
    t = np.linspace(0, 25, ns)
    fig, axs = plt.subplots(1, 2, figsize=(16, 6))
    labels = [r"$k_p$", r"$\tau_i$", r"$\tau_d$", r"$k_p$", r"$\tau_i$", r"$\tau_d$"]
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
  

    axs[0].plot(t, np.median(Ca_dat_EA, axis=1), color="tab:cyan", lw=1.5, label="CIRL (Extended)")
    axs[0].plot(t, np.median(Ca_eval_pid_lowop, axis=1), color="tab:blue", lw=1.5, label="CIRL (Initial)")
    axs[0].plot(
        t,
        np.median(Ca_eval_PID_const, axis=1),
        color="tab:orange",
        lw=1.5,
        label="Static PID",
    )

    axs[0].fill_between(
        t,
        np.min(Ca_dat_EA, axis=1),
        np.max(Ca_dat_EA, axis=1),
        color="tab:cyan",
        alpha=0.2,
        edgecolor="none",
    )

    axs[0].fill_between(
        t,
        np.min(Ca_eval_pid_lowop, axis=1),
        np.max(Ca_eval_pid_lowop, axis=1),
        color="tab:blue",
        alpha=0.2,
        edgecolor="none",
    )

    axs[0].fill_between(
        t,
        np.min(Ca_eval_PID_const, axis=1),
        np.max(Ca_eval_PID_const, axis=1),
        color="tab:orange",
        alpha=0.2,
        edgecolor="none",
    )
    axs[0].step(t, Ca_des, "--", lw=1.5, color="black")
    axs[0].set_ylabel("Concentration of B, $C_B$ (mol/m$^3$)")

    axs[0].legend(
        bbox_to_anchor=(0.5, 1.0),
        loc="lower center",
        ncol=3,
        frameon=False,
        columnspacing=0.5,
        handletextpad=0.5,
    )
    axs[0].set_xlim(min(t), max(t))

    axs[0].grid(True, alpha=0.5)
    axs[0].set_axisbelow(True)
    axs[0].set_xlabel("Time (min)")

    axs[1].step(
        t,
        np.median(Tc_dat_EA, axis=1),
        color="tab:cyan",
        linestyle="dashed",
        where="post",
        lw=1,
        label="CIRL (Extended)",
    )
    axs[1].step(
        t,
        np.median(Tc_eval_pid_lowop, axis=1),
        color="tab:blue",
        linestyle="dashed",
        where="post",
        lw=1,
        label="CIRL (Initial)",
    )
  
    axs[1].step(
        t,
        np.median(Tc_eval_PID_const, axis=1),
        color="tab:orange",
        linestyle="dashed",
        where="post",
        lw=1,
        label="Static PID",
    )
    axs[1].fill_between(
        t,
        np.min(Tc_dat_EA, axis=1),
        np.max(Tc_dat_EA, axis=1),
        color="tab:cyan",
        alpha=0.2,
    )
    axs[1].fill_between(
        t,
        np.min(Tc_eval_pid_lowop, axis=1),
        np.max(Tc_eval_pid_lowop, axis=1),
        color="tab:blue",
        alpha=0.2,
    )

    axs[1].fill_between(
        t,
        np.min(Tc_eval_PID_const, axis=1),
        np.max(Tc_eval_PID_const, axis=1),
        color="tab:orange",
        alpha=0.2,
    )
    axs[1].grid(True, alpha=0.5)
    axs[1].set_axisbelow(True)
    axs[1].set_ylabel("Cooling Temperature, $T_c$ (K)")
    axs[1].set_xlabel("Time (min)")

    axs[1].set_xlim(min(t), max(t))
    axs[1].legend(
        bbox_to_anchor=(0.5, 1.0),
        loc="lower center",
        ncol=3,
        frameon=False,
        columnspacing=0.5,
        handletextpad=0.5,
    )


    plt.subplots_adjust(wspace=0.3)
    plt.savefig("RLvsRLPID_states_highop.pdf")
    plt.show()

    fig, axs = plt.subplots(1, 1, figsize=(8, 6))
    axs.plot(
        t,
        np.median(ks_eval_pid_lowop[0, :, :], axis=1),
        col[0],
        lw=1.5,
        label="CIRL (Initial) " + labels[0],
    )
    axs.plot(
        t,
        np.median(ks_eval_EA[0, :, :], axis=1),
        col[0],
        lw=1.5,
        linestyle = 'dotted',
        label="CIRL (Extended) " + labels[0],
    )
    axs.step(
        t,
        np.median(ks_eval_pid_const[0, :, :], axis=1),
        color=col[0],
        linestyle="dashed",
        lw=1.5,
        label="Constant " + labels[0],
    )

    axs.fill_between(
        t,
        np.min(ks_eval_EA[0, :, :], axis=1),
        np.max(ks_eval_EA[0, :, :], axis=1),
        color=col_fill[0],
        alpha=0.2,
    )
    axs.fill_between(
        t,
        np.min(ks_eval_pid_lowop[0, :, :], axis=1),
        np.max(ks_eval_pid_lowop[0, :, :], axis=1),
        color=col_fill[0],
        alpha=0.2,
    )
    axs.set_ylabel("$C_B$-loop PID Gains")
    axs.set_xlabel("Time (min)")
    axs.set_xlim(min(t), max(t))
    axs.grid(True, alpha=0.5)
    axs.set_axisbelow(True)

    axs.plot(
        t,
        np.median(ks_eval_pid_lowop[1, :, :], axis=1),
        col[1],
        lw=1.5,
        label="CIRL (Initial) " + labels[1],
    )
    axs.plot(
        t,
        np.median(ks_eval_EA[1, :, :], axis=1),
        col[1],
        lw=1.5,
        linestyle = 'dotted',
        label="CIRL (Extended) " + labels[1],
    )
    axs.plot(
        t,
        np.median(ks_eval_pid_const[1, :, :], axis=1),
        color=col[1],
        linestyle="dashed",
        lw=1.5,
        label="Constant " + labels[1],
    )
  
    axs.fill_between(
        t,
        np.min(ks_eval_EA[1, :, :], axis=1),
        np.max(ks_eval_EA[1, :, :], axis=1),
        color=col_fill[1],
        alpha=0.2,
    )
 

    axs.plot(
        t,
        np.median(ks_eval_pid_lowop[2, :, :], axis=1),
        col[2],
        lw=1.5,
        label="CIRL (Initial) " + labels[2],
    )
    axs.plot(
        t,
        np.median(ks_eval_EA[2, :, :], axis=1),
        col[2],
        lw=1.5,
        linestyle = 'dotted',
        label="CIRL (Extended) " + labels[2],
    )
    axs.plot(
        t,
        np.median(ks_eval_pid_const[2, :, :], axis=1),
        color=col[2],
        linestyle="dashed",
        lw=1.5,
        label="Constant " + labels[2],
    )
   
    axs.fill_between(
        t,
        np.min(ks_eval_EA[2, :, :], axis=1),
        np.max(ks_eval_EA[2, :, :], axis=1),
        color=col_fill[2],
        alpha=0.2,
    )

    axs.fill_between(
        t,
        np.min(ks_eval_pid_lowop[2, :, :], axis=1),
        np.max(ks_eval_pid_lowop[2, :, :], axis=1),
        color=col_fill[2],
        alpha=0.2,
    )

    axs.legend(
        bbox_to_anchor=(0.5, 1.0),
        loc="lower center",
        ncol=3,
        frameon=False,
        columnspacing=0.5,
        handletextpad=0.5,
    )

   
    plt.tight_layout()
    plt.savefig("RLvsRLPIDsp_PID_highop.pdf")
    plt.show()

if __name__ == '__main__':
  env = reactor_class(test=True, ns=120, normRL=True)
  best_policy_rl_sd = torch.load('best_policy_rl_highop.pth')

  best_policy_pid_sd = torch.load('./data/best_policy_pid_highop.pth')


  env = reactor_class(test=True, ns=120,  normRL=False)
  best_policy_pid =cirl_net(
      n_fc1=16,
      n_fc2=16,
      activation=torch.nn.ReLU,
      n_layers=1,
      output_sz=6,
      input_sz=15,
      PID=True,
      deterministic=True,
  )
  best_policy_pid.load_state_dict(best_policy_pid_sd)
  Ca_eval_pid, T_eval_pid, V_eval_pid, Tc_eval_pid, F_eval_pid, ks_eval_pid = rollout(
      env, best_policy_pid, PID=True, ES=True
  )
  best_policy_pid_sd_lowop = torch.load('./data/best_policy_pid_lowop.pth')


  env = reactor_class(test=True, ns=120,  normRL=False)
  best_policy_pid = cirl_net(
      n_fc1=16,
      n_fc2=16,
      activation=torch.nn.ReLU,
      n_layers=1,
      output_sz=6,
      input_sz=15,

      PID=True,
      deterministic=True,
  )
  best_policy_pid.load_state_dict(best_policy_pid_sd_lowop)
  Ca_eval_pid_lowop, T_eval_pid_lowop, V_eval_pid_lowop, Tc_eval_pid_lowop, F_eval_pid_lowop, ks_eval_pid_lowop = rollout(
      env, best_policy_pid, PID=True, ES=True
  )



  env = reactor_class(test=True, ns=120,  normRL=False)
  best_policy_const_PID = np.load("./data/constant_gains_highop.npy")
  (
      Ca_eval_PID_const,
      T_eval_PID_const,
      V_eval_PID_const,
      Tc_eval_PID_const,
      F_eval_PID_const,
      ks_eval_pid_const,
  ) = rollout(env, best_policy_const_PID, PID=True, PG=False)



  SP = env.test_SP
  plot_simulation_comp(
      Ca_eval_pid,
      T_eval_pid,
      Tc_eval_pid,
      ks_eval_pid,
      F_eval_pid,
      V_eval_pid,
      Ca_eval_PID_const,
      T_eval_PID_const,
      V_eval_PID_const,
      Tc_eval_PID_const,
      F_eval_PID_const,
      ks_eval_pid_const,
      Ca_eval_pid_lowop,
      T_eval_pid_lowop,
      V_eval_pid_lowop,
      Tc_eval_pid_lowop,
      F_eval_pid_lowop,
      ks_eval_pid_lowop,
      SP,
      ns,
  )