import matplotlib.pyplot as plt
import numpy as np
from cstr_model import reactor_class


if __name__ == "__main__":
    n = 100
    Tc_vec = np.linspace(290, 450, n)
    F_vec = np.linspace(99, 110, n)
    ns = 120
    env = reactor_class(test=True, ns=ns, DS=True)

    F_j = 100
    x_b_slice = np.zeros(n)
    Tc_vec = np.linspace(290, 400, n)
    for i, Tc in enumerate(Tc_vec):
        env.reset()
        for ns_i in range(ns - 1):
            u = np.array([Tc, F_j])
            s_norm, r, done, info, _ = env.step(u)
            s = (
                s_norm * (env.observation_space.high - env.observation_space.low)
                + env.observation_space.low
            )
        x_b_slice[i] = env.state[1]
        x_b_slice[x_b_slice < 0] = 0

    SP1 = [0.7, 0.75, 0.8]
    SP2 = [0.4, 0.5, 0.6]
    SP3 = [0.1, 0.2, 0.3]
    Test = [0.075, 0.45, 0.725]

    plt.figure(figsize=(13, 10))
    plt.rcParams["text.usetex"] = "True"
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.size"] = 20

    plt.scatter(
        [Tc_vec[np.argmin(np.abs(x_b_slice - sp))] for sp in SP1],
        [x_b_slice[np.argmin(np.abs(x_b_slice - sp))] for sp in SP1],
        color="tab:red",
        label="Train Ep. 1",
        zorder=3,
        s=200,
    )
    plt.scatter(
        [Tc_vec[np.argmin(np.abs(x_b_slice - sp))] for sp in SP2],
        [x_b_slice[np.argmin(np.abs(x_b_slice - sp))] for sp in SP2],
        color="tab:green",
        label="Train Ep. 2",
        zorder=3,
        s=200,
    )
    plt.scatter(
        [Tc_vec[np.argmin(np.abs(x_b_slice - sp))] for sp in SP3],
        [x_b_slice[np.argmin(np.abs(x_b_slice - sp))] for sp in SP3],
        color="tab:purple",
        label="Train Ep. 3",
        zorder=3,
        s=200,
    )
    plt.scatter(
        [Tc_vec[np.argmin(np.abs(x_b_slice - test_i))] for test_i in Test],
        [x_b_slice[np.argmin(np.abs(x_b_slice - test_i))] for test_i in Test],
        marker="v",
        color="tab:orange",
        label="Test Episode",
        zorder=3,
        s=250,
        alpha=0.2,
    )
    plt.plot(Tc_vec, x_b_slice, color="tab:blue", zorder=2)
    y_min = [
        x_b_slice[np.argmin(np.abs(x_b_slice - sp))] for sp in [min(SP1 + SP2 + SP3)]
    ]
    y_max = [
        x_b_slice[np.argmin(np.abs(x_b_slice - sp))] for sp in [max(SP1 + SP2 + SP3)]
    ]

    # Fill the area between the lowest and highest set points
    plt.fill_between(
        Tc_vec,
        y_min,
        y_max,
        color="gray",
        alpha=0.3,
        label="Training Region",
        edgecolor="none",
        zorder=1,
    )
    plt.xlim(np.min(Tc_vec), np.max(Tc_vec))
    plt.ylim(0, 0.9)
    plt.xlabel(
        "Cooling Temperature, $T_C$",
    )
    plt.ylabel(
        "Concentration of species B, $C_B$",
    )
    plt.tick_params(
        axis="both",
        which="major",
    )
    # plt.grid(True)
    plt.legend(
        bbox_to_anchor=(0.5, 1.0),
        loc="lower center",
        ncol=5,
        frameon=False,
        columnspacing=0.5,
        handletextpad=0.5,
    )

    plt.savefig("..\\plots\\CS1_ds_training.pdf")
    plt.show()
