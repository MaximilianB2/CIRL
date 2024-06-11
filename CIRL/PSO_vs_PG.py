import torch
import torch.nn.functional as F
from cstr_model import reactor_class
from torch_pso import ParticleSwarmOptimizer
import copy
import numpy as np
import matplotlib.pyplot as plt
import pickle
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.callbacks import CheckpointCallback
from typing import Callable
from reward_callback_pg import LearningCurveCallback
def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """
    Linear learning rate schedule.

    :param initial_value: Initial learning rate.
    :return: schedule that computes
      current learning rate depending on remaining progress
    """
    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0.

        :param progress_remaining:
        :return: current learning rate
        """
        if progress_remaining < 0.7:
          return progress_remaining * initial_value
        elif progress_remaining < 0.3:
          return 0.3*initial_value
        else:
          return initial_value
    return func


# Train pure-rl
# log_file = "learning_curves_PG\PPO_RL_LC.csv"
# callback = LearningCurveCallback(log_file=log_file)
# env_normRL = reactor_class(test=False,ns = 120,normRL=True)
# model_normRL = PPO("MlPPOlicy", env_normRL, verbose=1,learning_rate=1e-2,seed=0,device = 'cuda')
# model_normRL.learn(int(5e4))
# model_normRL.save('PPO_normRL_3105')

training_reps = 10
# from stable_baselines3.common.callbacks import EvalCallback
# from stable_baselines3.common.monitor import Monitor

for i in range(3,training_reps):
  print(i)
  log_file = f"PPO_PID_LC_rep_{i}.csv"
  callback = LearningCurveCallback(log_file=log_file)
  policy_kwargs = dict(net_arch=dict(pi=[128, 128, 128]))
  env_PID = reactor_class(test=False,ns = 120,normRL=False)
  model_PID = PPO("MlpPolicy", env_PID, verbose=1,learning_rate=linear_schedule(1e-3),device = 'cpu',)

  # Create an evaluation environmenthttps://www.sciencedirect.com/science/article/pii/S0098135424001662


  # Create an evaluation callback
  # eval_callback = EvalCallback(eval_env, best_model_save_path='./logs/',
                              #  log_path='./logs/', eval_freq=5000, deterministic=True)

  model_PID.learn(int(1e6), callback=[callback])
  model_PID.save(f'PPO_PID_3105_rep_{i}')
