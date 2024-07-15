import torch
from cirl_policy import Net
from cstr_model import reactor_class
from torch_pso import ParticleSwarmOptimizer
import numpy as np
import pickle


class CIRLTraining:
    def __init__(
        self, n_fc_rl=128, n_fc_cirl=16, max_iter=150, training_reps=10
    ) -> None:
        self.n_fc_rl = n_fc_rl
        self.n_fc_cirl = n_fc_cirl
        self.max_iter = max_iter
        self.training_reps = training_reps

    @staticmethod
    def sample_uniform_params(params_prev, param_max, param_min):
        """Sample uniform parameters for policy initialization."""
        return {
            k: torch.rand(v.shape) * (param_max - param_min) + param_min
            for k, v in params_prev.items()
        }

    @staticmethod
    def criterion(policy, env):
        """Evaluate policy over multiple episodes and return average reward."""
        reps = 3
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
        return np.mean(ret)

    def training_loop(self, PID, dist_train, n_fc, highop=False):
        """Main training loop."""
        output_sz, normRL = (6, False) if PID else (2, True)
        global r_list, r_list_i, p_list
        r_list = []
        r_list_i = []
        p_list = []
        r_list_save = []
        p_list_save = []

        policy = Net(
            n_fc1=n_fc,
            n_fc2=n_fc,
            activation=torch.nn.ReLU,
            output_sz=output_sz,
            n_layers=1,
            input_sz=15,
            PID=True,
        )
        env = reactor_class(
            test=False,
            ns=120,
            normRL=normRL,
            dist=dist_train,
            dist_train=dist_train,
            highop=highop,
        )
        best_reward, best_policy = 1e8, None
        params = policy.state_dict()
        max_param, min_param = 0.1, -0.1

        for policy_i in range(30):  # Random Search
            NNparams_RS = self.sample_uniform_params(params, max_param, min_param)
            policy.load_state_dict(NNparams_RS)
            r = self.criterion(policy, env)
            if r < best_reward:
                best_policy, best_reward = policy, r
        print("Best reward after random search:", best_reward)
        print("PSO Algorithm...")
        best_reward = [best_reward]

        optim = ParticleSwarmOptimizer(
            policy.parameters(),
            inertial_weight=0.6,
            num_particles=30,
            max_param_value=max_param,
            min_param_value=min_param,
        )
        for i in range(self.max_iter):
            print(f"Iteration: {i+1} / {self.max_iter}")
            if i > 0:
                del r_list_i[:]

            def closure():
                # Clear any grads from before the optimization step, since we will be changing the parameters
                optim.zero_grad()
                return self.criterion(policy, env)

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
        return best_policy, r_list_save, p_list_save

    def sp_tracking_train(self, net_size_analysis=False):
        """Run the training process for the setpoint tracking example."""
        r_list_rl = []
        p_list_rl = []
        r_list_cirl = []
        p_list_cirl = []
        for _ in range(self.training_reps):
            # CIRL
            best_policy_cirl, r_cirl, p_cirl = self.training_loop(
                PID=True, dist_train=False, n_fc=self.n_fc_cirl
            )
            r_list_cirl.append(r_cirl)
            p_list_cirl.append(p_cirl)

            # RL
            best_policy_rl, r_rl, p_rl = self.training_loop(
                PID=False, dist_train=False, n_fc=self.n_fc_rl
            )
            r_list_rl.append(r_rl)
            p_list_rl.append(p_rl)

        if net_size_analysis:
            return (
                best_policy_rl,
                best_policy_cirl,
                r_list_cirl,
                r_list_rl,
                p_list_cirl,
                p_list_rl,
            )
        else:
            # Save rewards and policies
            torch.save(best_policy_rl.state_dict(), f"best_policy_rl_dist_{_}.pth")
            torch.save(best_policy_cirl.state_dict(), f"best_policy_pid_dist_{_}.pth")

            with open("r_pid.pkl", "wb") as f:
                pickle.dump(r_list_cirl, f)
            with open("r_rl.pkl", "wb") as f:
                pickle.dump(r_list_rl, f)
            with open("p_rl.pkl", "wb") as f:
                pickle.dump(p_list_rl, f)
            with open("p_pid.pkl", "wb") as f:
                pickle.dump(p_list_cirl, f)

    def sp_tracking_train_highop(self):
        """Run the training process for the highop setpoint tracking example."""

        r_list_cirl = []
        p_list_cirl = []
        for _ in range(self.training_reps):
            # CIRL
            best_policy_cirl, r_cirl, p_cirl = self.training_loop(
                PID=True, dist_train=False, n_fc=self.n_fc_cirl, highop=True
            )
            r_list_cirl.append(r_cirl)
            p_list_cirl.append(p_cirl)
            torch.save(best_policy_cirl.state_dict(), f"best_policy_pid_highop_{_}.pth")

        # Save rewards and policies
        with open("r_pid_highop.pkl", "wb") as f:
            pickle.dump(r_list_cirl, f)
        with open("p_pid_highop.pkl", "wb") as f:
            pickle.dump(p_list_cirl, f)

    def dist_tracking_train(
        self,
    ):
        """Run the training process for the disturbance rejection scenario."""
        r_list_rl = []
        p_list_rl = []
        r_list_cirl = []
        p_list_cirl = []
        for _ in range(self.training_reps):
            # CIRL
            best_policy_cirl, r_cirl, p_cirl = self.training_loop(
                PID=True, dist_train=True, n_fc=self.n_fc_cirl
            )
            r_list_cirl.append(r_cirl)
            p_list_cirl.append(p_cirl)
            torch.save(best_policy_cirl.state_dict(), f"best_policy_pid_dist_{_}.pth")

            # RL
            best_policy_rl, r_rl, p_rl = self.training_loop(
                PID=False, dist_train=True, n_fc=self.n_fc_rl
            )
            r_list_rl.append(r_rl)
            p_list_rl.append(p_rl)

            torch.save(best_policy_rl.state_dict(), f"best_policy_rl_dist_{_}.pth")

        # Save rewards and policies
        with open("r_pid_dist_nonobs.pkl", "wb") as f:
            pickle.dump(r_list_cirl, f)
        with open("r_rl_dist_nonobs.pkl", "wb") as f:
            pickle.dump(r_list_rl, f)
        with open("p_rl_dist_nonobs.pkl", "wb") as f:
            pickle.dump(p_list_rl, f)
        with open("p_pid_dist_nonobs.pkl", "wb") as f:
            pickle.dump(p_list_cirl, f)


if __name__ == "__main__":
    trainer = CIRLTraining()
    trainer.sp_tracking_train()
    trainer.dist_tracking_train()
    trainer.sp_tracking_train_highop()
