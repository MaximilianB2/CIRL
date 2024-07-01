import torch
import torch.nn.functional as F
from cstr_model import reactor_class
from torch_pso import ParticleSwarmOptimizer
import copy
import numpy as np
import matplotlib.pyplot as plt
from cirl_policy import Net as cirl_net
ns = 120


def sample_uniform_params(params_prev, param_max, param_min):
    params = {
        k: torch.rand(v.shape) * (param_max - param_min) + param_min
        for k, v in params_prev.items()
    }
    return params


def sample_local_params(params_prev, param_max, param_min):
    params = {
        k: torch.rand(v.shape) * (param_max - param_min) + param_min + v
        for k, v in params_prev.items()
    }
    return params


class Net(torch.nn.Module):
    def __init__(
        self, n_fc1, n_fc2, activation, n_layers, input_sz, output_sz, PID, **kwargs
    ):
        super(Net, self).__init__()

        # Unpack the dictionary
        self.args = kwargs
        self.dtype = torch.float
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device("cpu")
        self.pid = PID
        self.input_size = input_sz  # State size: Ca, T,Ca-,T- , Ca setpoint
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
        # y = F.tanh(out) #[-1,1]
        if self.pid:
            y = F.tanh(mu)  # [
        else:
            y = torch.clamp(mu, -1, 1)
        y = y.detach().numpy()
        return y


def criterion(policy, env):
    reps = 1
    ret = np.zeros(reps)
    for i in range(reps):
        s, _ = env.reset()
        rew = 0
        while True:
            a = policy(torch.tensor(s)).detach().numpy()
            s, r, done, _, _ = env.step(a)
            # r = -r #PSO minimises
            rew += r
            # except:
            #   done = True
            #   rew += 1e5

            if done:
                break
        ret[i] = rew

    r_tot = np.mean(ret)
    global r_list
    global p_list
    global r_list_i
    r_list.append(r_tot)
    r_list_i.append(r_tot)
    p_list.append(policy)
    return r_tot


def training_loop(PID, max_iter):
    if PID:
        print("Training with PID Controller")
        output_sz = 6
        normRL = False
    else:
        print("Training without PID Controller")
        output_sz = 2
        normRL = True

    policy = Net(
        n_fc1=256,
        n_fc2=256,
        activation=torch.nn.ReLU,
        output_sz=output_sz,
        n_layers=1,
        PID=PID,
    )
    env = reactor_class(test=False, ns=120, normRL=normRL, dist=True)
    # Training Loop Parameters
    new_swarm = 0
    best_reward = 1e8
    i = 0
    global r_list, r_list_i, p_list
    r_list = []
    r_list_i = []
    p_list = []
    r_list_save = []
    p_list_save = []
    evals_rs = 30

    params = policy.state_dict()
    # Random Search
    max_param = 0.1
    min_param = max_param * -1
    print("Random search to find good initial policy...")
    for policy_i in range(evals_rs):
        # sample a random policy
        NNparams_RS = sample_uniform_params(params, max_param, min_param)
        # consruct policy to be evaluated
        policy.load_state_dict(NNparams_RS)
        # evaluate policy
        r = criterion(policy, env)
        # Store rewards and policies
        if r < best_reward:
            best_policy = p_list[r_list.index(r)]
            best_reward = r
            init_params = copy.deepcopy(NNparams_RS)
    policy.load_state_dict(init_params)
    # PSO Optimisation paramters
    optim = ParticleSwarmOptimizer(
        policy.parameters(),
        inertial_weight=0.6,
        num_particles=10,
        max_param_value=max_param,
        min_param_value=min_param,
    )
    print("Best reward after random search:", best_reward)
    print("PSO Algorithm...")
    best_reward = [1e8]
    while i < max_iter:
        print(f"Iteration: {i+1} / {max_iter}")
        if i > 0:
            del r_list_i[:]
        # if i > 0 and abs(best_reward[-1]-best_reward[-2]) < tol:
        #  break

        def closure():
            # Clear any grads from before the optimization step, since we will be changing the parameters
            optim.zero_grad()
            return criterion(policy, env)

        optim.step(closure)
        new_swarm = min(r_list_i)
        r_list_save.append(new_swarm)
        p_list_save.append(p_list[r_list.index(new_swarm)].state_dict())
        if new_swarm < best_reward[-1]:
            best_reward.append(new_swarm)
            best_policy = p_list[r_list.index(new_swarm)]

            print(
                f"New best reward: {best_reward[-1]} ({best_reward[-1]/3}) per training episode"
            )

        i += 1
    print("Finished optimisation")
    print("Best reward:", best_reward[-1])
    return best_policy, r_list_save, p_list_save


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
            Ks_norm = ((a_policy.detach().numpy() + 1) / 2) * (x_norm[1] - x_norm[0]) + x_norm[0]
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
                
                Ks_norm = ((a_policy.detach().numpy() + 1) / 2) * (x_norm[1] - x_norm[0]) + x_norm[0]
                ks_eval_EA[:, i, r_i] = Ks_norm
            try:
                s_norm, r, done, info, _ = env.step(a_policy)
            except:
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
        print(r_eval_EA)
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
    Ca_eval_PID_PG,
    T_eval_PID_PG,
    V_eval_PID_PG,
    Tc_eval_PID_PG,
    F_eval_PID_PG,
    Ca_eval_normRL_PG,
    T_eval_normRL_PG,
    V_eval_normRL_PG,
    Tc_eval_normRL_PG,
    F_eval_normRL_PG,
    ks_obs,
    SP,
    ns,
):
    plt.rcParams["text.usetex"] = "True"
    plt.rcParams["font.family"] = "serif"
    t = np.linspace(0, 25, ns)
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
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
    axs[0].plot(
        t, np.median(Ca_dat_PG, axis=1), color="tab:red", lw=1, label="RL (non-obs)"
    )
    axs[0].plot(
        t,
        np.median(Ca_dat_EA, axis=1),
        color="tab:blue",
        lw=1.5,
        label="CIRL (non-obs)",
    )
    # axs[0].plot(t,np.median(Ca_eval_PID_PG,axis=1), color = 'tab:orange', lw=1.5, label = 'CIRL (obs)')
    # axs[0].plot(t,np.median(Ca_eval_normRL_PG,axis=1), color = 'tab:green', lw=1.5, label = 'RL (obs)')
    # axs[0].plot(t,np.median(Ca_dat_GS,axis=1), color = 'tab:orange', lw=1.5, label = 'Offline GS')
    axs[0].fill_between(
        t,
        np.min(Ca_dat_PG, axis=1),
        np.max(Ca_dat_PG, axis=1),
        color="tab:red",
        alpha=0.2,
    )
    axs[0].fill_between(
        t,
        np.min(Ca_dat_EA, axis=1),
        np.max(Ca_dat_EA, axis=1),
        color="tab:blue",
        alpha=0.2,
    )
    # axs[0].fill_between(t, np.min(Ca_eval_normRL_PG,axis=1), np.max(Ca_eval_normRL_PG,axis=1),color = 'tab:green', alpha=0.2)
    # axs[0].fill_between(t, np.min(Ca_eval_PID_PG,axis=1), np.max(Ca_eval_PID_PG,axis=1),color = 'tab:orange', alpha=0.2)
    axs[0].step(t, Ca_des, "--", lw=1.5, color="black")
    axs[0].set_ylabel("Concentration of B, $C_B$ (mol/m$^3$)")
    axs[0].set_xlabel("Time (min)")
    axs[0].legend(loc="best")
    axs[0].set_xlim(min(t), max(t))
    axs[0].grid(True, alpha=0.5)
    axs[0].set_axisbelow(True)
    axs[0].set_xlabel("Time (min)")
    # axs[0].set_ylim(0.6,1)

    # axs[1].plot(t, np.median(T_dat_PG,axis=1), color = 'tab:red', lw=1,label = 'RL (non-obs)')
    # axs[1].plot(t,np.median(T_dat_EA,axis=1), color = 'tab:blue', lw=1.5, label = 'CIRL (non-obs)')
    # axs[1].plot(t,np.median(T_eval_normRL_PG,axis=1), color = 'tab:green', lw=1.5, label = 'RL (obs)')
    # axs[1].plot(t,np.median(T_eval_PID_PG,axis=1), color = 'tab:orange', lw=1.5, label = 'CIRL (obs)')
    # #axs[1].fill_between(t, np.min(T_dat_PG,axis=1), np.max(T_dat_PG,axis=1),color = 'tab:red', alpha=0.1,edgecolor  = 'none')
    # axs[1].fill_between(t, np.min(T_dat_EA,axis=1), np.max(T_dat_EA,axis=1),color = 'tab:blue', alpha=0.2,edgecolor  = 'none')
    # axs[1].fill_between(t, np.min(T_eval_normRL_PG,axis=1), np.max(T_eval_normRL_PG,axis=1),color = 'tab:green', alpha=0.2,edgecolor  = 'none')
    # axs[1].fill_between(t, np.min(T_eval_PID_PG,axis=1), np.max(T_eval_PID_PG,axis=1),color = 'tab:orange', alpha=0.2,edgecolor  = 'none')
    # axs[1].set_ylabel('Temperature (K))')
    # axs[1].set_xlabel('Time (min)')
    # axs[1].legend(loc='best')
    # axs[1].set_xlim(min(t), max(t))
    # axs[1].grid(True, alpha = 0.5)
    # axs[1].set_axisbelow(True)
    # axs[1].set_ylim(325,340)

    # axs[1].plot(t, np.median(V_eval_PG,axis=1), color = 'tab:red', lw=1, label = 'RL (non-obs)')
    # axs[1].plot(t, np.median(V_eval_EA,axis=1), color = 'tab:blue', lw=1, label = 'CIRL (non-obs)')
    # # axs[1].plot(t, np.median(V_eval_normRL_PG, axis=1), color = 'tab:green', lw=1, label = 'RL (obs)')
    # # axs[1].plot(t, np.median(V_eval_PID_PG,axis=1), color = 'tab:orange', lw=1, label = 'CIRL (obs)')
    # axs[1].fill_between(t, np.min(V_eval_PG,axis=1), np.max(V_eval_PG,axis=1),color = 'tab:red', alpha=0.2,edgecolor  = 'none')
    # axs[1].fill_between(t, np.min(V_eval_EA,axis=1), np.max(V_eval_EA,axis=1),color = 'tab:blue', alpha=0.2,edgecolor  = 'none')
    # # axs[1].fill_between(t, np.min(V_eval_normRL_PG,axis=1), np.max(V_eval_normRL_PG,axis=1),color = 'tab:green', alpha=0.2,edgecolor  = 'none')
    # # axs[1].fill_between(t, np.min(V_eval_PID_PG,axis=1), np.max(V_eval_PID_PG,axis=1),color = 'tab:orange', alpha=0.2,edgecolor  = 'none')
    # axs[1].step(t, V_des, '--', lw=1.5, color='black')
    # axs[1].set_ylabel('Volume, $V$ (m$^3$)')
    # axs[1].set_xlabel('Time (min)')
    # axs[1].legend(loc='best')
    # axs[1].grid(True, alpha = 0.5)
    # axs[1].set_xlim(min(t), max(t))
    # axs[1].set_xlabel('Time (min)')
    # axs[1].set_ylim(95,110)

    axs[1].step(
        t,
        np.median(Tc_dat_PG, axis=1),
        color="tab:red",
        linestyle="dashed",
        where="post",
        lw=1,
        label="RL (non-obs)",
    )
    axs[1].step(
        t,
        np.median(Tc_dat_EA, axis=1),
        color="tab:blue",
        linestyle="dashed",
        where="post",
        lw=1,
        label="CIRL (non-obs)",
    )
    # axs[1].step(t, np.median(Tc_eval_normRL_PG,axis=1), color = 'tab:green', linestyle = 'dashed', where= 'post',lw=1, label = 'RL (obs)')
    # axs[1].step(t, np.median(Tc_eval_PID_PG,axis=1), color = 'tab:orange', linestyle = 'dashed', where= 'post',lw=1, label = 'CIRL (obs)')
    axs[1].fill_between(
        t,
        np.min(Tc_dat_PG, axis=1),
        np.max(Tc_dat_PG, axis=1),
        color="tab:red",
        alpha=0.2,
        edgecolor="none",
    )
    axs[1].fill_between(
        t,
        np.min(Tc_dat_EA, axis=1),
        np.max(Tc_dat_EA, axis=1),
        color="tab:blue",
        alpha=0.2,
        edgecolor="none",
    )

    axs[1].grid(True, alpha=0.5)
    axs[1].set_axisbelow(True)
    axs[1].set_ylabel("Cooling Temperature, $T_c$ (K)")
    axs[1].set_xlabel("Time (min)")
    axs[1].legend(loc="best")
    axs[1].set_xlim(min(t), max(t))

    # axs[3].step(t, np.median(F_eval_PG,axis=1), color = 'tab:red', linestyle = 'dashed' , where = 'post',lw=1, label = 'RL (non-obs)')
    # axs[3].step(t, np.median(F_eval_EA,axis=1), color = 'tab:blue', linestyle = 'dashed', where= 'post',lw=1, label = 'CIRL (non-obs)')
    # axs[1].fill_between(t, np.min(F_eval_PG,axis=1), np.max(F_eval_PG,axis=1),color = 'tab:red', alpha=0.2,edgecolor  = 'none')
    # axs[1].fill_between(t, np.min(F_eval_EA,axis=1), np.max(F_eval_EA,axis=1),color = 'tab:blue', alpha=0.2,edgecolor  = 'none')
    # # axs[3].step(t, np.median(F_eval_normRL_PG,axis=1), color = 'tab:green', linestyle = 'dashed', where= 'post',lw=1, label = 'RL (obs)')
    # # axs[3].step(t, np.median(F_eval_PID_PG,axis=1), color = 'tab:orange', linestyle = 'dashed', where= 'post',lw=1, label = 'CIRL (obs)')
    # axs[3].grid(True, alpha = 0.5)
    # axs[3].set_xlim(min(t), max(t))
    # axs[3].set_ylabel('Flowrate in, $F_{in}$ (m$^3$/s)')
    # axs[3].set_xlabel('Time (min)')
    # axs[3].legend(loc='best')
    plt.subplots_adjust(wspace=0.3)
    plt.savefig("disturbance_train.pdf")
    plt.show()

    fig, axs = plt.subplots(1, 3, figsize=(20, 7))

    axs[0].plot(
        t,
        np.median(ks_eval_EA[0, :, :], axis=1),
        col[0],
        lw=1,
        label="CIRL (non-obs) " + labels[0],
    )
    # axs[0].plot(t, np.median(ks_obs[0,:,:],axis=1), color = 'gold', lw=1,label = 'CIRL (obs) ' + labels[0])
    # axs[0].plot(t, np.median(ks_eval_const[0,:,:],axis=1), color = 'black',linestyle = 'dashed', lw=1,label = 'Constant ' + labels[0])
    # axs[0].plot(t, np.median(ks_eval_GS[0,:,:],axis=1), col[0],linestyle = 'dashed', lw=1.5,label = 'GS ' + labels[0])
    axs[0].fill_between(
        t,
        np.min(ks_eval_EA[0, :, :], axis=1),
        np.max(ks_eval_EA[0, :, :], axis=1),
        color=col_fill[0],
        alpha=0.2,
    )
    axs[0].fill_between(
        t,
        np.min(ks_obs[0, :, :], axis=1),
        np.max(ks_obs[0, :, :], axis=1),
        color="gold",
        alpha=0.2,
    )
    axs[0].set_ylabel("$C_B$-loop $K_p$")
    axs[0].set_xlabel("Time (min)")
    axs[0].legend(loc="best")
    axs[0].set_xlim(min(t), max(t))
    axs[0].grid(True, alpha=0.5)
    axs[0].set_axisbelow(True)

    axs[1].plot(
        t,
        np.median(ks_eval_EA[1, :, :], axis=1),
        col[1],
        lw=1.5,
        label="CIRL (non-obs) " + labels[1],
    )
    # axs[1].plot(t, np.median(ks_obs[1,:,:],axis=1), color = 'brown', lw=1.5,label = 'CIRL (obs) ' + labels[1])
    # axs[1].plot(t, np.median(ks_eval_const[1,:,:],axis=1), color = 'black',linestyle = 'dashed', lw=1.5, label = 'Constant ' + labels[1])
    # axs[1].plot(t, np.median(ks_eval_GS[1,:,:],axis=1), col[1],linestyle = 'dashed', lw=1.5, label = 'GS ' + labels[1])
    axs[1].fill_between(
        t,
        np.min(ks_eval_EA[1, :, :], axis=1),
        np.max(ks_eval_EA[1, :, :], axis=1),
        color=col_fill[1],
        alpha=0.2,
    )
    axs[1].fill_between(
        t,
        np.min(ks_obs[1, :, :], axis=1),
        np.max(ks_obs[1, :, :], axis=1),
        color="brown",
        alpha=0.2,
    )
    axs[1].set_ylabel("$C_B$-loop $K_i$")
    axs[1].set_xlabel("Time (min)")
    axs[1].legend(loc="upper left")
    axs[1].set_xlim(min(t), max(t))
    axs[1].grid(True, alpha=0.5)
    axs[1].set_axisbelow(True)

    axs[2].plot(
        t,
        np.median(ks_eval_EA[2, :, :], axis=1),
        col[2],
        lw=2,
        label="CIRL (non-obs) " + labels[2],
    )
    # axs[2].plot(t, np.median(ks_obs[2,:,:],axis=1), color = 'tab:cyan', lw=2,label ='CIRL (obs) ' + labels[2])
    # axs[2].plot(t, np.median(ks_eval_const[2,:,:],axis=1), color = 'black',linestyle = 'dashed', lw=2, label = 'Constant ' + labels[2])
    # axs[2].plot(t, np.median(ks_eval_GS[2,:,:],axis=1), col[2],linestyle = 'dashed', lw=2, label = 'GS ' + labels[2])
    axs[2].fill_between(
        t,
        np.min(ks_eval_EA[2, :, :], axis=1),
        np.max(ks_eval_EA[2, :, :], axis=1),
        color=col_fill[2],
        alpha=0.2,
    )
    axs[2].fill_between(
        t,
        np.min(ks_obs[2, :, :], axis=1),
        np.max(ks_obs[2, :, :], axis=1),
        color="tab:cyan",
        alpha=0.2,
    )
    axs[2].set_ylabel("$C_B$-loop $K_d$")
    axs[2].set_xlabel("Time (min)")
    axs[2].legend(loc="best")
    axs[2].set_xlim(min(t), max(t))
    axs[2].grid(True, alpha=0.5)
    axs[2].set_axisbelow(True)
    plt.savefig("RLvsRLPIDdist_PID_CA.pdf")
    plt.show()

    fig, axs = plt.subplots(1, 3, figsize=(20, 7))
    axs[0].set_title("EA PID Parameters")

    axs[0].plot(
        t,
        np.median(ks_eval_EA[3, :, :], axis=1),
        col[0],
        lw=1,
        label="CIRL (Non-obs) " + labels[0],
    )
    axs[0].plot(
        t,
        np.median(ks_obs[3, :, :], axis=1),
        color="gold",
        lw=1,
        label="CIRL (obs) " + labels[0],
    )
    # axs[0].plot(t, np.median(ks_eval_const[0,:,:],axis=1), color = 'black',linestyle = 'dashed', lw=1,label = 'Constant ' + labels[0])
    # axs[0].plot(t, np.median(ks_eval_GS[0,:,:],axis=1), col[0],linestyle = 'dashed', lw=1.5,label = 'GS ' + labels[0])
    axs[0].fill_between(
        t,
        np.min(ks_eval_EA[3, :, :], axis=1),
        np.max(ks_eval_EA[3, :, :], axis=1),
        color=col_fill[0],
        alpha=0.2,
    )
    axs[0].fill_between(
        t,
        np.min(ks_obs[3, :, :], axis=1),
        np.max(ks_obs[3, :, :], axis=1),
        color="gold",
        alpha=0.2,
    )
    axs[0].set_ylabel("$V$-loop $K_p$")
    axs[0].set_xlabel("Time (min)")
    axs[0].legend(loc="best")
    axs[0].set_xlim(min(t), max(t))
    axs[0].grid(True, alpha=0.5)
    axs[0].set_axisbelow(True)

    axs[1].plot(
        t,
        np.median(ks_eval_EA[4, :, :], axis=1),
        col[1],
        lw=1.5,
        label="CIRL (non-obs) " + labels[1],
    )
    axs[1].plot(
        t,
        np.median(ks_obs[4, :, :], axis=1),
        color="brown",
        lw=1.5,
        label="CIRL, (obs) " + labels[1],
    )
    # axs[1].plot(t, np.median(ks_eval_const[1,:,:],axis=1), color = 'black',linestyle = 'dashed', lw=1.5, label = 'Constant ' + labels[1])
    # axs[1].plot(t, np.median(ks_eval_GS[1,:,:],axis=1), col[1],linestyle = 'dashed', lw=1.5, label = 'GS ' + labels[1])
    axs[1].fill_between(
        t,
        np.min(ks_eval_EA[4, :, :], axis=1),
        np.max(ks_eval_EA[4, :, :], axis=1),
        color=col_fill[1],
        alpha=0.2,
    )
    axs[1].fill_between(
        t,
        np.min(ks_obs[4, :, :], axis=1),
        np.max(ks_obs[4, :, :], axis=1),
        color="brown",
        alpha=0.2,
    )
    axs[1].set_ylabel("$V$-loop $K_i$")
    axs[1].set_xlabel("Time (min)")
    axs[1].legend(loc="upper left")
    axs[1].set_xlim(min(t), max(t))
    axs[1].grid(True, alpha=0.5)
    axs[1].set_axisbelow(True)

    axs[2].plot(
        t,
        np.median(ks_eval_EA[5, :, :], axis=1),
        col[2],
        lw=2,
        label="CIRL (non-obs) " + labels[2],
    )
    axs[2].plot(
        t,
        np.median(ks_obs[5, :, :], axis=1),
        color="tab:cyan",
        lw=2,
        label="CIRL (obs) " + labels[2],
    )
    # axs[2].step(t, np.median(ks_eval_const[2,:,:],axis=1), color = 'black',linestyle = 'dashed',where = 'post', lw=2, label = 'Constant ' + labels[2])
    # axs[2].step(t, np.median(ks_eval_GS[2,:,:],axis=1), col[2],linestyle = 'dashed',where = 'post', lw=2, label = 'GS ' + labels[2])
    axs[2].fill_between(
        t,
        np.min(ks_eval_EA[5, :, :], axis=1),
        np.max(ks_eval_EA[5, :, :], axis=1),
        color=col_fill[2],
        alpha=0.2,
    )
    axs[2].fill_between(
        t,
        np.min(ks_obs[5, :, :], axis=1),
        np.max(ks_obs[5, :, :], axis=1),
        color="tab:cyan",
        alpha=0.2,
    )
    axs[2].set_ylabel("$V$-loop  $K_d$")
    axs[2].set_xlabel("Time (min)")
    axs[2].legend(loc="best")
    axs[2].set_xlim(min(t), max(t))
    axs[2].grid(True, alpha=0.5)
    axs[2].set_axisbelow(True)
    plt.savefig("RLvsRLPIDdist_PID_V.pdf")
    plt.show()


# env = reactor_class(test = True, ns = 120, PID_vel= True, normRL=True,dist=True)
# best_policy_rl = Net(n_fc1 = 128,n_fc2 = 128,activation = torch.nn.ReLU,n_layers = 1,output_sz=2,deterministic=True)
# best_policy_rl.load_state_dict(torch.load('best_policy_rl_wnoise.pth'))
# Ca_eval_RL,T_eval_RL,V_eval_RL,Tc_eval_RL,F_eval_RL = rollout(env,best_policy_rl,PID=False)


# env = reactor_class(test = True, ns = 120, PID_vel= True, normRL=False,dist=True)
# best_policy_pid = Net(n_fc1 = 128,n_fc2 = 128,activation = torch.nn.ReLU,n_layers = 1,output_sz=6,deterministic=True)
# best_policy_pid.load_state_dict(torch.load('best_policy_pid_wnoise.pth'))
# Ca_eval_pid,T_eval_pid,V_eval_pid,Tc_eval_pid,F_eval_pid,ks_eval_pid = rollout(env,best_policy_pid,PID=True)

# env = reactor_class(test = True, ns = 120, PID_vel= True, normRL=False,dist=True)
# best_policy_PG_PID = SAC.load('SAC_PID_1604.zip')
# Ca_eval_PID_PG,T_eval_PID_PG,V_eval_PID_PG,Tc_eval_PID_PG,F_eval_PID_PG,ks_eval_pid_PG = rollout(env,best_policy_PG_PID,PID=True,PG=True)

# env = reactor_class(test = True, ns = 120, PID_vel= True, normRL=True,dist=True)
# best_policy_PG_normRL = SAC.load('SAC_normRL_1604.zip')
# Ca_eval_normRL_PG,T_eval_normRL_PG,V_eval_normRL_PG,Tc_eval_normRL_PG,F_eval_normRL_PG = rollout(env,best_policy_PG_normRL,PID=False,PG=True)

# SP = env.test_SP
# plot_simulation_comp(Ca_eval_RL, T_eval_RL, Tc_eval_RL,Ca_eval_pid, T_eval_pid, Tc_eval_pid,ks_eval_pid,F_eval_pid,F_eval_RL,V_eval_pid,V_eval_RL,Ca_eval_PID_PG,T_eval_PID_PG,V_eval_PID_PG,Tc_eval_PID_PG,F_eval_PID_PG,Ca_eval_normRL_PG,T_eval_normRL_PG,V_eval_normRL_PG,Tc_eval_normRL_PG,F_eval_normRL_PG,SP,ns)


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
best_policy_rl.load_state_dict(torch.load("best_policy_rl_dist.pth"))
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
best_policy_pid.load_state_dict(torch.load("best_policy_pid_dist.pth"))
Ca_eval_pid, T_eval_pid, V_eval_pid, Tc_eval_pid, F_eval_pid, ks_eval_pid = rollout(
    env, best_policy_pid, PID=True
)

# env = reactor_class(
#     test=True, ns=120,  normRL=False, dist=True, dist_obs=True
# )
# best_policy_PID_obs = Net(
#     n_fc1=128,
#     n_fc2=128,
#     activation=torch.nn.ReLU,
#     n_layers=1,
#     output_sz=6,
#     input_sz=9,
#     deterministic=True,
#     PID=True,
# )
# best_policy_PID_obs.load_state_dict(torch.load("best_policy_pid_distobs.pth"))
# (
#     Ca_eval_PID_PG,
#     T_eval_PID_PG,
#     V_eval_PID_PG,
#     Tc_eval_PID_PG,
#     F_eval_PID_PG,
#     ks_eval_pid_PG,
# ) = rollout(env, best_policy_PID_obs, PID=True, PG=False)

# env = reactor_class(
#     test=True, ns=120,  normRL=True, dist=True, dist_obs=True
# )
# best_policy_rl_obs = Net(
#     n_fc1=128,
#     n_fc2=128,
#     activation=torch.nn.ReLU,
#     n_layers=1,
#     output_sz=2,
#     input_sz=9,
#     deterministic=True,
#     PID=True,
# )
# best_policy_rl_obs.load_state_dict(torch.load("best_policy_rl_distobs.pth"))
# (
#     Ca_eval_normRL_PG,
#     T_eval_normRL_PG,
#     V_eval_normRL_PG,
#     Tc_eval_normRL_PG,
#     F_eval_normRL_PG,
# ) = rollout(env, best_policy_rl_obs, PID=False, PG=False)
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
