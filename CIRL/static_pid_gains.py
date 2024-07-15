import numpy as np
from cstr_model import reactor_class
from scipy.optimize import differential_evolution

# Global variables for the simulation
ns = 120 * 3
reps = 1  # Number of repetitions for the simulation
env = reactor_class(test=True, ns=ns)  # Environment instance


def initialize_evaluation_matrices(ns, reps):
    """
    Initializes matrices for storing simulation results.

    Parameters:
    - ns: int, number of simulation steps.
    - reps: int, number of repetitions.

    Returns:
    - A tuple of initialized numpy arrays for Ca_eval, T_eval, Tc_eval, V_eval, F_eval, ks_eval, r_eval.
    """
    Ca_eval = np.zeros((ns, reps))
    T_eval = np.zeros((ns, reps))
    Tc_eval = np.zeros((ns, reps))
    V_eval = np.zeros((ns, reps))
    F_eval = np.zeros((ns, reps))
    ks_eval = np.zeros((6, ns, reps))
    r_eval = np.zeros((1, reps))
    return Ca_eval, T_eval, Tc_eval, V_eval, F_eval, ks_eval, r_eval


def rollout(Ks, reps, env, ns):
    """
    Simulates the reactor environment with given PID gains and evaluates performance.

    Parameters:
    - Ks: array, PID gains.
    - reps: int, number of repetitions for averaging performance.
    - env: gym env, instance of the environment
    - ns: int, number of steps to simulate

    Returns:
    - The negative mean reward
    """

    # Initialize evaluation matrices
    Ca_eval, T_eval, Tc_eval, V_eval, F_eval, ks_eval, r_eval = (
        initialize_evaluation_matrices(ns, reps)
    )

    x_norm = env.x_norm

    for r_i in range(reps):
        s_norm, _ = env.reset()
        s = (
            s_norm * (env.observation_space.high - env.observation_space.low)
            + env.observation_space.low
        )
        Ca_eval[0, r_i] = s[0]
        T_eval[0, r_i] = s[1]
        V_eval[0, r_i] = s[2]
        Tc_eval[0, r_i] = 300.0  # Initial temperature control
        F_eval[0, r_i] = 100  # Initial flow rate
        Ks_norm = ((Ks[:6] + 1) / 2) * (x_norm[1] - x_norm[0]) + x_norm[0]
        ks_eval[:, 0, r_i] = Ks_norm
        r_tot = 0

        for i in range(1, ns):
            s_norm, r, done, _, info = env.step(Ks[:6])
            ks_eval[:, i, r_i] = info["Ks"]
            r_tot += r
            s = (
                s_norm * (env.observation_space.high - env.observation_space.low)
                + env.observation_space.low
            )
            Ca_eval[i, r_i] = s[0]
            T_eval[i, r_i] = s[1]
            V_eval[i, r_i] = s[2]

        r_eval[:, r_i] = r_tot

    r = np.mean(r_eval, axis=1)

    return r


if __name__ == "__main__":
    # Optimization bounds for the PID gains
    bounds = [(-1, 1)] * 6

    # Perform differential evolution to find the optimal PID gains
    result_vel = differential_evolution(
        rollout,
        bounds=bounds,
        args=(1, env, ns),
        maxiter=150,
        disp=True,
    )

    # Save the optimal PID gains to a file
    np.save("..\\data\\constant_gains.npy", result_vel.x)
