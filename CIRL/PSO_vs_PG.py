import torch
import torch.nn.functional as F
from cstr_model import reactor_class
from torch_pso import ParticleSwarmOptimizer
import copy
import numpy as np
import matplotlib.pyplot as plt
import pickle
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.callbacks import CheckpointCallback
from typing import Callable
from reward_callback_pg import LearningCurveCallback


# 
log_file = "learning_curves_PG\SAC_RL_LC.csv"
callback = LearningCurveCallback(log_file=log_file)
env_normRL = reactor_class(test=False,ns = 120,PID_vel=True,normRL=True)
model_normRL = SAC("MlpPolicy", env_normRL, verbose=1,learning_rate=1e-2,seed=0,device = 'cuda')
model_normRL.learn(int(5e4))
model_normRL.save('SAC_normRL_3105')

log_file = "learning_curves_PG\SAC_PID_LC.csv"
callback = LearningCurveCallback(log_file=log_file)
env_PID = reactor_class(test=False,ns = 120,PID_vel=True,normRL=False)
model_PID = SAC("MlpPolicy", env_PID, verbose=1,learning_rate=1e-2,seed=0,device = 'cuda')
model_PID.learn(int(5e4))
model_PID.save('SAC_PID_3105')
