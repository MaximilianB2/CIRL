import gymnasium as gym
from gymnasium import spaces
import numpy as np
from scipy.integrate import odeint
import copy

def PID_velocity(Ks,e,e_history,u_prev,ts,s_hist):
    '''
    Dynamic velocity form PID 
    
    '''
    
    # time step
    dt = ts[1] - ts[0]

    # PID parameters
    KpCa = Ks[0]; KiCa = Ks[1] + 1e-6; KdCa = Ks[2] # add epsilon to stop division by 0 
    KpF = Ks[3]; KiF = Ks[4] + 1e-6; KdF = Ks[5] 
   
    # Cooling temp
    Tc = u_prev[-1][0] + KpCa*(e[0] - e_history[-1,0]) + (KpCa/KiCa)*e[0]*dt - KpCa*KdCa*(e[0]-2*e_history[-1,0]+e_history[-2,0])/dt
    Tc = min(max(Tc,290),450) # Clamp between operational limits

    # Flow in
    F = u_prev[-1][1] + KpF*(e[1] - e_history[-1,1]) + (KpF/KiF)*e[1]*dt - KpF*KdF*(e[1]-2*e_history[-1,1]+e_history[-2,1])/dt
    F = min(max(F,99),105) 

    u = np.array([Tc,F])
    return u


def cstr_CS1(x,t,u,Tf,Caf):
    '''
    Dynamic model of cstr with volume and concentration control.
  
    '''
    # ==  Inputs (2) == #
    Tc  = u[0] # Temperature of Cooling Jacket (K)
    Fin = u[1] # Volumetric Flowrate at inlet (m^3/sec) = 100
  
    # == States (5) == #
    Ca = x[0] # Concentration of A in CSTR (mol/m^3)
    Cb = x[1] # Concentration of B in CSTR (mol/m^3)
    Cc = x[2] # Concentration of C in CSTR (mol/m^3)
    T  = x[3] # Temperature in CSTR (K)
    V  = x[4] # Volume in CSTR (K)

    # == Process parameters == #
    Tf       = 350    # Feed Temperature (K)
    Caf      = Caf      # Feed Concentration of A (mol/m^3)
    Fout     = 100    # Volumetric Flowrate at outlet (m^3/sec)
    #V       = 100    # Volume of CSTR (m^3)
    rho      = 1000   # Density of A-B Mixture (kg/m^3)
    Cp       = 0.239  # Heat Capacity of A-B-C Mixture (J/kg-K)
    UA       = 5e4    # U -Heat Transfer Coefficient (W/m^2-K) A -Area - (m^2)
    # Reaction A->B
    mdelH_AB  = 5e3    # Heat of Reaction for A->B (J/mol)
    EoverR_AB = 8750   # E -Activation Energy (J/mol), R -Constant = 8.31451 J/mol-K
    k0_AB     = 7.2e10 # Pre-exponential Factor for A->B (1/sec)#
    rA        = k0_AB*np.exp(-EoverR_AB/T)*Ca # reaction rate
    # Reaction B->C
    mdelH_BC  = 4e3      # Heat of Reaction for B->C (J/mol) => 5e4
    EoverR_BC = 10750    # E -Activation Energy (J/mol), R -Constant = 8.31451 J/mol-K !! 10
    k0_BC     = 8.2e10   # Pre-exponential Factor for A->B (1/sec)# !! 8
    rB        = k0_BC*np.exp(-EoverR_BC/T)*Cb # reaction rate !! **2
    # play with mdelH_BC, factor on Cb**2 and k0_BC, maybe even EoverR_BC

    # == Concentration Derivatives == #
    dCadt    = (Fin*Caf - Fout*Ca)/V - rA  # A Concentration Derivative
    dCbdt    = rA - rB - Fout*Cb/V         # B Concentration Derivative
    dCcdt    = rB      - Fout*Cc/V         # B Concentration Derivative
    dTdt     = Fin/V*(Tf - T) \
              + mdelH_AB/(rho*Cp)*rA \
              + mdelH_BC/(rho*Cp)*rB \
              + UA/V/rho/Cp*(Tc-T)   # Calculate temperature derivative
    dVdt     = Fin - Fout

    # == Return xdot == #
    
    xdot    = np.zeros(5)
    xdot[0] = dCadt
    xdot[1] = dCbdt
    xdot[2] = dCcdt
    xdot[3] = dTdt
    xdot[4] = dVdt
    return xdot


# Create a gym environment
class reactor_class(gym.Env):
  '''
  Gym environment of cstr case study
  
  '''
  def __init__(self,ns,test = False, DS = False,normRL = False, dist = False, dist_train = False, dist_obs = False):
    
    # Import environment parameters
    self.DS = DS
    self.dist = dist
    self.dist_train = dist_train
    self.dist_obs = dist_obs
    self.ns = ns 
    self.test = test
    self.normRL = normRL

    
    # Setpoints
    Ca_des1 = [0.70 for i in range(int(ns/3))] + [0.75 for i in range(int(ns/3))] + [0.86 for i in range(int(ns/3))]
    Ca_des2 = [0.1 for i in range(int(ns/3))] + [0.2 for i in range(int(ns/3))] + [0.3 for i in range(int(ns/3))]
    Ca_des3 = [0.4 for i in range(int(ns/3))] + [0.5 for i in range(int(ns/3))] + [0.6 for i in range(int(ns/3))]
   
   
    
    if self.test:
      # Ca_des1 = [0.075 for i in range(int(ns/3))] + [0.45 for i in range(int(ns/3))] + [0.9 for i in range(int(ns/3))] 
      Ca_des1 = [0.075 for i in range(int(ns/3))] + [0.45 for i in range(int(ns/3))] + [0.725 for i in range(int(ns/3))] 
    V_des = [100 for i in range(int(ns/3))] + [100 for i in range(int(ns/3))] + [100 for i in range(int(ns/3))] 

    if self.dist:
      Ca_des1 = Ca_des2 = Ca_des3 = [0.4 for i in range(int(ns))] 

    self.test_SP = np.array([Ca_des1,V_des])
    self.SP = np.array(([[Ca_des1],[Ca_des2],[Ca_des3]],[[V_des]]),dtype = object)

    # Training disturbances
    self.Caf_dist = np.array([1.7,1.6,1.9]) 

    # Obs and action spaces
    self.x_norm = np.array(([0,0,0.01,0,0,0.01,],[25,20,10,1,2,1])) 
    # self.x_norm = np.array(([-5,0,0.01,-1,0,0.01,],[25,20,10,1,2,1])) # PID space
    self.observation_space = spaces.Box(low = np.array([0, 350,90,0, 350,90,0,99]),high= np.array([1,390,102,1,390,103,1,101]))#Cb,T,V,Cb,T,V,Ca_sp,V_sp
    if self.dist_obs:
      self.observation_space = spaces.Box(low = np.array([0, 350,90,0, 350,90,0,99,1]),high= np.array([1,390,102,1,390,103,1,101,2]))#Cb,T,V,Cb,T,V,Ca_sp,V_sp
    if self.normRL:
      self.action_space = spaces.Box(low = np.array([-1]*2),high= np.array([1]*2)) 
    else:
      self.action_space = spaces.Box(low = np.array([-1]*6),high= np.array([1]*6))

  
    # Initial points
    self.Ca_ss = 0.80
    self.T_ss  = 327
    self.V_ss = 102
    self.x0    = np.empty(2)
    self.x0[0] = self.Ca_ss
    self.x0[1] = self.T_ss
  
    self.u_ss = 300.0 # Steady State Initial Condition

    self.Caf  = 1     # Feed Concentration (mol/m^3)

    # Time Interval (min)
    self.t = np.linspace(0,100,ns)

    # Store results for plotting
    self.Ca = np.ones(len(self.t)) * self.Ca_ss
    self.T  = np.ones(len(self.t)) * self.T_ss 
    self.Tf = 350 # Feed Temperature (K)

  

  def reset(self, seed = None):
    '''
    Reset environment back to initial state

    '''

    self.i = 0
    self.SP_i = 0
    self.Caf = 1
    Ca_des = self.SP[0][self.SP_i][0][0]
    V_des = self.SP[1][0][0][0]
    self.state = np.array([self.Ca_ss,0,0,self.T_ss,self.V_ss,self.Ca_ss,0,0,self.T_ss,self.V_ss,Ca_des,V_des])
    self.done = False


    if not self.test:
      self.disturb = False
    
    self.u_history = []
    self.e_history = []
    self.s_history = []

    self.ts = [self.t[self.i],self.t[self.i+1]]
    self.RL_state = [self.state[i] for i in [1,3,4,6,8,9,10,11]]
    
    if self.dist_obs:
      self.RL_state.append(self.Caf)   

    self.state_norm = (self.RL_state - self.observation_space.low)/(self.observation_space.high - self.observation_space.low)
    
    return self.state_norm,{}

  def step(self, action_policy):
    '''
    Take a step in the environment
    '''
    if self.DS:
       self.u_DS = action_policy
       self.info = {}

    self.action = action_policy
    Ca_des = self.SP[0][self.SP_i][0][self.i]
    V_des = self.SP[1][0][0][self.i]
    self.state, rew = self.reactor(self.state,self.action,Ca_des,V_des)
    self.i += 1

    if self.i == self.ns:
        if self.test:
          self.done = True
        elif self.SP_i < 2:
          self.Caf = 1
          self.SP_i += 1
          self.i = 0
          Ca_des = self.SP[0][self.SP_i][0][0]
          V_des = self.SP[1][0][0][0]
          self.state = np.array([self.Ca_ss,0,0,self.T_ss,self.V_ss,self.Ca_ss,0,0,self.T_ss,self.V_ss,Ca_des,V_des])
          self.u_history = []
          self.e_history = []
        else:
          self.done = True

    self.RL_state = [self.state[i] for i in [1,3,4,6,8,9,10,11]]
    if self.dist_obs:
      self.RL_state.append(self.Caf)   
    self.state_norm = (self.RL_state - self.observation_space.low)/(self.observation_space.high - self.observation_space.low)
    return self.state_norm,rew,self.done,False,self.info

  def reactor(self,state,action,Ca_des,V_des):
    '''
    Integrate the environment and calculate reward.
    '''
    Ca = state[0]
    Cb = state[1]
    Cc = state[2]
    T  = state[3]
    V  = state[4]

    x_sp    = np.array([Ca_des,V_des])
    e = np.zeros(2)
    e[0]  = x_sp[0] - state[1]
    e[1]  = x_sp[1] - state[4]
          

    Ks = action #Ca, T, u, Ca setpoint and T setpoint
    
    if not self.DS and not self.normRL:
      try:
        
        Ks_norm = ((Ks.detach().numpy() + 1) / 2) * (self.x_norm[1] - self.x_norm[0]) + self.x_norm[0]
        
      except Exception:
        
        Ks_norm = ((Ks + 1) / 2) * (self.x_norm[1] - self.x_norm[0]) + self.x_norm[0]

      self.info = {'Ks':Ks_norm}
      
      
      if self.i < 2:
          u = np.array([302,99])
      else:
          u =  PID_velocity(Ks_norm,e,np.array(self.e_history),self.u_history,self.ts,np.array(self.s_history))
    if self.DS:
      u = self.u_DS

    # simulate system
    u_max = np.array([400,105])
    u_min = np.array([290,99])
    if self.normRL:
      if self.i < 2:
        u = np.array([302,99])
      else:
        try:
          u = ((action + 1) / 2) * (u_max - u_min) + u_min
        except Exception:
          u = ((action[0] + 1) / 2) * (u_max - u_min) + u_min
      self.info = {'u': u}
  
    if self.i > 70 and self.dist:
      if self.dist_train:
        self.Caf = self.Caf_dist[self.SP_i]
      else:
        self.Caf = 1.75
    
    y       = odeint(cstr_CS1,state[0:5],self.ts,args=(u,self.Tf,self.Caf))

    # add measurement noise
    Ca_plus = y[-1][0] + np.random.uniform(low=-0.001,high=0.001)
    Cb_plus = y[-1][1] + np.random.uniform(low=-0.001,high=0.001)
    Cc_plus = y[-1][2] + np.random.uniform(low=-0.001,high=0.001)
    T_plus  = y[-1][3] + np.random.uniform(low=-.1,high=0.1)
    V_plus = y[-1][4]  + np.random.uniform(low=-.01,high=0.01)

    # collect data
    state_plus = np.zeros(12)
    state_plus[0]   = Ca_plus
    state_plus[1]   = Cb_plus
    state_plus[2]   = Cc_plus
    state_plus[3]   = T_plus
    state_plus[4] = V_plus
    state_plus[5]   = Ca
    state_plus[6]   = Cb
    state_plus[7]   = Cc
    state_plus[8]   = T
    state_plus[9]   = V
    state_plus[10]   = Ca_des
    state_plus[11]  = V_des


    self.e_history.append((e))
    if self.i == 0:
      u_cha = np.zeros(2)
    else:
      u_cha = (u-self.u_history[-1])**2
    
   
    self.u_history.append(u)
    self.s_history.append(state[0:2])
    
    r_x = 0
    r_x = (e[0])**2
    r_x +=(e[1])**2 / 10
    r_x += 0.0005*u_cha[0] + 0.005*u_cha[1]
 
    
    return state_plus, r_x