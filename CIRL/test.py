import numpy as np 
from cstr_model import reactor_class

reactor = reactor_class(ns = 120, )


norm_gains = np.load('..\\data\\constant_gains.npy')

Ks_norm = ((norm_gains + 1) / 2) * (
                  reactor.x_norm[1] - reactor.x_norm[0]
              ) + reactor.x_norm[0]

print(Ks_norm)