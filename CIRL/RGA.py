from cstr_model import reactor_class
import numpy as np
import matplotlib.pyplot as plt

# Initialize the CSTR environment
cstr = reactor_class(ns = 120, test=True, DS= True)

# Define input and output variables
inputs = ['Cooling Temperature', 'Inlet Flow Rate']
outputs = ['Volume', 'Concentration of B']

def perform_step_test(mv_index, step_size=0.1, steps=120):
    # Reset the environment
    state, _ = cstr.reset()
    
    # Apply step change to the specified MV
    action = np.array([302,100])
    if mv_index == 0:
      step_size = 10
      action[mv_index] += step_size
    else:
      step_size = 10
      action[mv_index] += step_size
    
    # Run simulation
    states = [state]
    for i in range(steps):
        if i < 60:
            next_state, _, _, _, _ = cstr.step(np.array([302,100]))
        else:
            next_state, _, _, _, _ = cstr.step(action)
        states.append(next_state)

    states = np.array(states)
    states  = (
                states * (cstr.observation_space.high - cstr.observation_space.low)
                + cstr.observation_space.low
            )
            

    # Calculate gains
    initial_state = states[60]
    final_state = states[-1]
    all_gains = (final_state - initial_state) / step_size
    gains = np.array([all_gains[2], all_gains[0]])
    
    return gains, states

def plot_step_test(states, mv_index):
    plt.figure(figsize=(12, 8))
    
    # Plot only from timestep 30 onwards
    plot_states = states[30:]
    time_steps = np.arange(len(plot_states))
    
    plt.subplot(2, 1, 1)
    plt.plot(time_steps, plot_states[:, 2], label='Volume')
    plt.axvline(x=30, color='r', linestyle='--', label='Step Change')
    plt.title(f'Step Test for {inputs[mv_index]}')
    plt.ylabel('Volume')
    plt.legend()
    
    plt.subplot(2, 1, 2)
    plt.plot(time_steps, plot_states[:, 0], label='Concentration of B')
    plt.axvline(x=30, color='r', linestyle='--', label='Step Change')
    plt.xlabel('Time Step')
    plt.ylabel('Concentration of B')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

def perform_rga_analysis(num_runs=1, plot=False):
    K_total = np.zeros((2, 2))
    
    for run in range(num_runs):
        K = np.zeros((2, 2))
        for i in range(2):
            gains, states = perform_step_test(i)
            K[:, i] = gains
            
            if plot and run == 0:  # Plot only for the first run
                plot_step_test(states, i)
        
        K_total += K
    
    # Calculate average K
    K_avg = K_total / num_runs
    
    # Calculate RGA
    H = K_avg * np.linalg.inv(K_avg).T
    RGA = H / np.sum(H, axis=1, keepdims=True)
    
    return RGA

# Perform multiple RGA analyses and average the results
num_analyses = 3
RGA_total = np.zeros((2, 2))

for i in range(num_analyses):
    RGA_total += perform_rga_analysis(plot=(i==0))  # Plot only for the first analysis

RGA_final = RGA_total / num_analyses

print("\nRGA in LaTeX format:")
print("\\begin{bmatrix}")
print(f"{RGA_final[0,0]:.4f} & {RGA_final[0,1]:.4f} \\\\")
print(f"{RGA_final[1,0]:.4f} & {RGA_final[1,1]:.4f}")
print("\\end{bmatrix}")

print("Final Relative Gain Array (averaged over multiple analyses):")
print(RGA_final)

# Interpret results
pairings = [(0, 0), (1, 1)] if RGA_final[0, 0] > 0.5 else [(0, 1), (1, 0)]

print("\nRecommended pairings:")
for i, j in pairings:
    print(f"{inputs[i]} -> {outputs[j]}")







    