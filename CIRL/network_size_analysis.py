import pickle
from training_CIRL import CIRLTraining

if __name__ == "__main__":
    neurons = [16, 128]
    results_rl_network = []
    results_pid_network = []
    for t_i in range(10):
        for i, neuron_i in enumerate(neurons):
            trainer_i = CIRLTraining(
                n_fc_cirl=neuron_i, n_fc_rl=neuron_i, training_reps=1
            )
            (
                best_policy_rl,
                best_policy_cirl,
                r_list_cirl,
                r_list_rl,
                p_list_cirl,
                p_list_rl,
            ) = trainer_i.sp_tracking_train(net_size_analysis=True)
            results_rl_network[i] = {
                "r_list": r_list_rl,
                "p_list": p_list_rl,
                "best_policy": best_policy_rl,
            }
            results_pid_network[i] = {
                "r_list": r_list_cirl,
                "p_list": p_list_cirl,
                "best_policy": best_policy_cirl,
            }
        # Save results
        with open(f"results_pid_network_rep_newobs_{t_i}.pkl", "wb") as f:
            pickle.dump(results_pid_network, f)

        with open(f"results_rl_network_rep_newobs_{t_i}.pkl", "wb") as f:
            pickle.dump(results_rl_network, f)
