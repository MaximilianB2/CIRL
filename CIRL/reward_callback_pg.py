from stable_baselines3.common.callbacks import BaseCallback
import csv

class LearningCurveCallback(BaseCallback):
    def __init__(self, verbose=0, log_file="learning_curve.csv"):
        super(LearningCurveCallback, self).__init__(verbose)
        self.episode_rewards = []
        self.log_file = log_file

    def _on_step(self):
        return True

    def _on_rollout_end(self):
        # Record episode rewards at the end of each episode
        if len(self.model.ep_info_buffer) > 0 and self.model.ep_info_buffer[-1] is not None:
            self.episode_rewards.append(self.model.ep_info_buffer[-1].get('r', 0.0))

    def _on_training_end(self):
        # Save rewards to CSV file
        with open(self.log_file, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Episode', 'Reward'])
            for i, reward in enumerate(self.episode_rewards):
                writer.writerow([i, reward])