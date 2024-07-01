import torch
from cirl_policy import Net
from cstr_model import reactor_class
from torch_pso import ParticleSwarmOptimizer
import copy
import numpy as np
import pickle


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


def criterion(policy, env):
    reps = 1
    ret = np.zeros(reps)
    for i in range(reps):
        s, _ = env.reset()
        rew = 0
        while True:
            a = policy(torch.tensor(s)).detach().numpy()

            s, r, done, _, _ = env.step(a)

            rew += r

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


def training_loop(PID, max_iter, dist_train, dist_obs, prev_policy=None):
    if PID:
        print("Training with PID Controller")
        output_sz = 6
        normRL = False
    else:
        print("Training without PID Controller")
        output_sz = 2
        normRL = True
    if dist_obs:
        input_sz = 9
    else:
        input_sz = 15
    initialise = False
    if prev_policy is not None:
        initialise = True

    policy = Net(
        n_fc1=128,
        n_fc2=128,
        activation=torch.nn.ReLU,
        output_sz=output_sz,
        n_layers=1,
        input_sz=input_sz,
        PID=True,
    )
    env = reactor_class(
        test=True,
        ns=120,
        normRL=normRL,
        dist=dist_train,
        dist_obs=dist_obs,
        dist_train=dist_train,
    )
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
    if prev_policy is not None:
        evals_rs = 1
    params = policy.state_dict()
    # Random Search
    max_param = 0.2  # SP 0.01
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
    if initialise:
        policy.load_state_dict(prev_policy)
        r = criterion(policy, env)
        best_reward = r
    else:
        policy.load_state_dict(init_params)
    # PSO Optimisation paramters
    optim = ParticleSwarmOptimizer(
        policy.parameters(),
        inertial_weight=0.6,  # 0.7 for PID
        num_particles=30,  # 10 for PID
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
        r_list_save.append(np.array(r_list_i))

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


max_iter = 75
training_reps = 1
r_pid = []
r_rl = []
p_pid = []
p_rl = []

# for r_i in range(training_reps):
#     # print(f"Training repition: {r_i +1}")
#     best_policy_pid, r_list_pid, p_list_pid = training_loop(PID = True,max_iter = max_iter, dist_obs=False, dist_train=False)
#     r_pid.append(r_list_pid)
#     p_pid.append(p_list_pid)
#     # torch.save(best_policy_pid.state_dict(),'best_policy_pid_unstable.pth')
#     # np.save('rewards_PID.npy',r_list_pid)

#     best_policy_rl, r_list_rl, p_list_rl = training_loop(
#         PID=False, max_iter=max_iter, dist_obs=False, dist_train=False
#     )

#     r_rl.append(r_list_rl)

#     p_rl.append(p_list_rl)
#     #   # np.save('rewards_rl.npy',r_list_rl,)
#     # torch.save(best_policy_rl.state_dict(), "best_policy_rl.pth")


# with open('r_pid.pkl', 'wb') as f:
#     pickle.dump(r_pid, f)
# with open("r_rl.pkl", "wb") as f:
#     pickle.dump(r_rl, f)
# with open("p_rl.pkl", "wb") as f:
#     pickle.dump(p_rl, f)
# with open('p_pid.pkl', 'wb') as f:
#     pickle.dump(p_pid, f)
# plt.figure()
# plt.plot(r_list_pid[1:], label='PID')
# plt.plot(r_list_rl[1:], label='RL')
# plt.legend()
# plt.show()

# # Train with disturbance not observed
r_pid_dist_nonobs = []
# r_rl_dist_nonobs = []
p_pid_dist_nonobs = []
# p_rl_dist_nonobs = []
for r_i in range(training_reps):
    print(f'Training repition: {r_i +1}')
    #   print('PID with no disturbance observation')
    # best_policy_pid, r_list_pid, p_list_pid = training_loop(PID = True,max_iter = max_iter,dist_train=True,dist_obs=False)
    #   r_pid_dist_nonobs.append(r_list_pid)
    #   p_pid_dist_nonobs.append(p_list_pid)
    # torch.save(best_policy_pid.state_dict(),'best_policy_pid_dist.pth')
    #   # np.save('rewards_PID.npy',r_list_pid)

    best_policy_rl, r_list_rl, p_list_rl = training_loop(PID = False,max_iter = max_iter,dist_train=True,dist_obs=False)
    # r_rl_dist_nonobs.append(r_list_rl)
    # p_rl_dist_nonobs.append(p_list_rl)
    # np.save('rewards_rl.npy',r_list_rl,)
    torch.save(best_policy_rl.state_dict(),'best_policy_rl_dist.pth')

# with open('r_pid_dist_nonobs.pkl', 'wb') as f:
#     pickle.dump(r_pid_dist_nonobs, f)
# # with open('r_rl_dist_nonobs.pkl', 'wb') as f:
# #     pickle.dump(r_rl_dist_nonobs, f)
# # with open('p_rl_dist_nonobs.pkl', 'wb') as f:
# #     pickle.dump(p_rl_dist_nonobs, f)
# with open('p_pid_dist_nonobs.pkl', 'wb') as f:
#     pickle.dump(p_pid_dist_nonobs, f)

# # # # Train with disturbance observed
# # r_pid_dist_obs = []
# r_rl_dist_obs = []
# # p_pid_dist_obs = []
# p_rl_dist_obs = []
# for r_i in range(training_reps):
# #   print(f'Training repition: {r_i +1}')
# #   print('PID with disturbance observation')
# #   best_policy_pid, r_list_pid, p_list_pid = training_loop(PID = True,max_iter = max_iter, dist_train=True, dist_obs=True)
# #   r_pid_dist_obs.append(r_list_pid)
# #   p_pid_dist_obs.append(p_list_pid)
# #   torch.save(best_policy_pid.state_dict(),'best_policy_pid_distobs.pth')

#   best_policy_rl, r_list_rl, p_list_rl = training_loop(PID = False,max_iter = max_iter, dist_train=True, dist_obs=True)
#   r_rl_dist_obs.append(r_list_rl)
#   p_rl_dist_obs.append(p_list_rl)

#   torch.save(best_policy_rl.state_dict(),'best_policy_rl_distobs.pth')


# with open('r_pid_dist_obs.pkl', 'wb') as f:
#     pickle.dump(r_pid_dist_obs, f)
# with open('r_rl_dist_obs.pkl', 'wb') as f:
#     pickle.dump(r_rl_dist_obs, f)
# with open('p_rl_dist_obs.pkl', 'wb') as f:
#     pickle.dump(p_rl_dist_obs, f)
# # with open('p_pid_dist_obs.pkl', 'wb') as f:
#     pickle.dump(p_pid_dist_obs, f)
