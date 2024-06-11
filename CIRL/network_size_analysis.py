import torch
import torch.nn.functional as F
from cstr_model import reactor_class
from torch_pso import ParticleSwarmOptimizer
import copy
import numpy as np
import matplotlib.pyplot as plt
import pickle
def sample_uniform_params(params_prev, param_max, param_min):
    params = {k: torch.rand(v.shape)* (param_max - param_min) + param_min \
              for k, v in params_prev.items()}
    return params

def sample_local_params(params_prev, param_max, param_min):
    params = {k: torch.rand(v.shape)* (param_max - param_min) + param_min + v \
              for k, v in params_prev.items()}
    return params

class Net(torch.nn.Module):

  def __init__(self, n_fc1, n_fc2, activation,n_layers,input_sz,output_sz,PID,**kwargs):
    super(Net, self).__init__()

    # Unpack the dictionary
    self.args     = kwargs
    self.dtype    = torch.float
    self.use_cuda = torch.cuda.is_available()
    self.device   = torch.device("cpu")
    self.pid = PID
    self.input_size = input_sz #State size: Ca, T,Ca-,T- , Ca setpoint 
    self.output_sz  = output_sz #Output size: Reactor Ks size
    self.n_layers = torch.nn.ModuleList()
    self.hs1        = n_fc1                                    # !! parameters
    self.hs2        = n_fc2                                      # !! parameter

    # defining layer
    self.hidden1 = torch.nn.Linear(self.input_size, self.hs1,bias=True)
    self.act = activation()
    self.hidden2 = torch.nn.Linear(self.hs1, self.hs2,bias=True)
    for i in range(0,n_layers):
      linear_layer = torch.nn.Linear(self.hs2,self.hs2)
      self.n_layers.append(linear_layer)
    self.output_mu = torch.nn.Linear(self.hs2, self.output_sz, bias=True)
    self.output_std = torch.nn.Linear(self.hs2, self.output_sz, bias=True)

  def forward(self, x):

    x = x.float()
    y           = self.act(self.hidden1(x))
    y           = self.act(self.hidden2(y))        
    mu = self.output_mu(y)
    # log_std = self.output_std(y) 
    # dist = torch.distributions.Normal(mu, log_std.exp() + 1e-6)
    # out = dist.sample()  
    out = mu                                       
    # y = F.tanh(out) #[-1,1]
  
    if self.pid:
      y = F.tanh(out) #[-1,1]
    else:
      y = torch.clamp(out,-1,1)
    return y

def criterion(policy,env):
  reps = 1
  ret = np.zeros(reps)
  for i in range(reps):
    s, _  = env.reset()
    rew = 0
    while True:
      a = policy(torch.tensor(s)).detach().numpy()

      s, r, done,_,_ = env.step(a)

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


def training_loop(PID,max_iter,dist_train, dist_obs,prev_policy = None, fc_neurons = 128):
  if PID:
     print('Training with PID Controller')
     output_sz = 6
     normRL = False
  else:
     print('Training without PID Controller')
     output_sz = 2
     normRL = True
  if dist_obs:
     input_sz = 9
  else:
     input_sz = 8 
  initialise = False
  if prev_policy is not None:
     initialise = True
 
  policy = Net(n_fc1 = fc_neurons,n_fc2 = fc_neurons,activation = torch.nn.ReLU,output_sz=output_sz,n_layers = 1,input_sz=input_sz,PID=True)
  env = reactor_class(test = False, ns = 120,  normRL=normRL,dist = dist_train, dist_obs=dist_obs, dist_train=dist_train)
  #Training Loop Parameters
  new_swarm = 0
  best_reward = 1e8
  i = 0
  global r_list, r_list_i, p_list
  r_list = []
  r_list_i = []
  p_list  = []
  r_list_save = []
  p_list_save = []
  evals_rs = 30
  if prev_policy is not None:
     evals_rs = 1
  params = policy.state_dict()
  #Random Search
  max_param = 0.05#SP 0.01
  min_param = max_param*-1
  print('Random search to find good initial policy...')
  for policy_i in range(evals_rs):
      # sample a random policy
      NNparams_RS  = sample_uniform_params(params, max_param, min_param)
      # consruct policy to be evaluated
      policy.load_state_dict(NNparams_RS)
      # evaluate policy
      r = criterion(policy,env)
      #Store rewards and policies
      if r < best_reward:
          best_policy = p_list[r_list.index(r)]
          best_reward = r
          init_params= copy.deepcopy(NNparams_RS)
  if initialise:
    policy.load_state_dict(prev_policy)
    r =  criterion(policy,env)
    best_reward = r
  else:
    policy.load_state_dict(init_params)
  #PSO Optimisation paramters
  optim = ParticleSwarmOptimizer(policy.parameters(),
                                inertial_weight=0.6, #0.7 for PID
                                num_particles=30, #10 for PID
                                max_param_value=max_param,
                                min_param_value=min_param)
  print('Best reward after random search:', best_reward)
  print('PSO Algorithm...')
  best_reward = [1e8]

  while i < max_iter :
      print(f'Iteration: {i+1} / {max_iter}')
      if i > 0:
        del r_list_i[:]
      #if i > 0 and abs(best_reward[-1]-best_reward[-2]) < tol:
        #  break
      
     
      def closure():
          # Clear any grads from before the optimization step, since we will be changing the parameters
          optim.zero_grad()
          return criterion(policy,env)
      optim.step(closure)
      new_swarm = min(r_list_i)
      r_list_save.append(np.array(r_list_i))

      p_list_save.append(p_list[r_list.index(new_swarm)].state_dict())
      if new_swarm < best_reward[-1]:
        best_reward.append(new_swarm)
        best_policy = p_list[r_list.index(new_swarm)]
        
        print(f'New best reward: {best_reward[-1]} ({best_reward[-1]/3}) per training episode')
      
      i += 1
  print('Finished optimisation')
  print('Best reward:', best_reward[-1]) 
  return best_policy, r_list_save, p_list_save

max_iter = 75

results_pid_network = {}
results_rl_network = {}
training_repition = 3
neurons = [8,16,32,64,128,256]

for i in range(training_repition):
  print(f'Training Rep: {i}')
  for r_i in range(len(neurons)):
    print(f'Neurons: {neurons[r_i]}')
    best_policy_pid, r_list_pid, p_list_pid = training_loop(PID = True,max_iter = max_iter, dist_obs=False, dist_train=False, fc_neurons=neurons[r_i])
    
    results_pid_network[r_i] = {
        'r_list': r_list_pid,
        'p_list': p_list_pid,
        'best_policy': best_policy_pid.state_dict()
    }
    
    torch.save(best_policy_pid.state_dict(),f'best_policy_pid_{str(neurons[r_i])}_neurons_rep_{i}.pth')

    best_policy_rl, r_list_rl, p_list_rl = training_loop(PID = False,max_iter = max_iter,dist_obs=False, dist_train=False,fc_neurons=neurons[r_i])
    
    results_rl_network[r_i] = {
        'r_list': r_list_rl,
        'p_list': p_list_rl,
        'best_policy': best_policy_rl.state_dict()
    }
    
    torch.save(best_policy_rl.state_dict(),f'best_policy_rl_{str(neurons[r_i])}_neurons_rep_{i}.pth')


    with open(f'results_pid_network_rep_{i}.pkl', 'wb') as f:
        pickle.dump(results_pid_network, f)

    with open(f'results_rl_network_rep_{i}.pkl', 'wb') as f:
        pickle.dump(results_rl_network, f)




