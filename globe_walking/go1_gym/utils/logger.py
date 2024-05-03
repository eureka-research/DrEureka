import torch
import numpy as np

class Logger:
    def __init__(self, env):
        self.env = env

    def populate_log(self, env_ids):

        extras = {}

        # fill extras
        if len(env_ids) > 0:
            extras["train/episode"] = {}
            for key in self.env.episode_sums.keys():
                extras["train/episode"]['rew_' + key] = torch.mean(
                    self.env.episode_sums[key][env_ids])
                self.env.episode_sums[key][env_ids] = 0.
            extras["train/episode"]["episode_length"] = self.env.episode_length_buf[env_ids]

        # log additional curriculum info
        if self.env.cfg.terrain.curriculum:
            extras["train/episode"]["terrain_level"] = torch.mean(
                self.env.terrain_levels.float())
        extras["env_bins"] = torch.Tensor(self.env.env_command_bins)
        # if self.env.cfg.commands.command_curriculum:
        #     commands = self.env.commands
        #     extras["train/episode"]["min_command_duration"] = torch.min(commands[:, 8])
        #     extras["train/episode"]["max_command_duration"] = torch.max(commands[:, 8])
        #     extras["train/episode"]["min_command_bound"] = torch.min(commands[:, 7])
        #     extras["train/episode"]["max_command_bound"] = torch.max(commands[:, 7])
        #     extras["train/episode"]["min_command_offset"] = torch.min(commands[:, 6])
        #     extras["train/episode"]["max_command_offset"] = torch.max(commands[:, 6])
        #     extras["train/episode"]["min_command_phase"] = torch.min(commands[:, 5])
        #     extras["train/episode"]["max_command_phase"] = torch.max(commands[:, 5])
        #     extras["train/episode"]["min_command_freq"] = torch.min(commands[:, 4])
        #     extras["train/episode"]["max_command_freq"] = torch.max(commands[:, 4])
        #     extras["train/episode"]["min_command_x_vel"] = torch.min(commands[:, 0])
        #     extras["train/episode"]["max_command_x_vel"] = torch.max(commands[:, 0])
        #     extras["train/episode"]["min_command_y_vel"] = torch.min(commands[:, 1])
        #     extras["train/episode"]["max_command_y_vel"] = torch.max(commands[:, 1])
        #     extras["train/episode"]["min_command_yaw_vel"] = torch.min(commands[:, 2])
        #     extras["train/episode"]["max_command_yaw_vel"] = torch.max(commands[:, 2])
        #     # if self.env.cfg.commands.num_commands > 9:
        #     extras["train/episode"]["min_command_swing_height"] = torch.min(commands[:, 9])
        #     extras["train/episode"]["max_command_swing_height"] = torch.max(commands[:, 9])
        #     for curriculum, category in zip(self.env.curricula, self.env.category_names):
        #         extras["train/episode"][f"command_area_{category}"] = np.sum(curriculum.weights) / \
        #                                                                    curriculum.weights.shape[0]

        #     extras["train/episode"]["min_action"] = torch.min(self.env.actions)
        #     extras["train/episode"]["max_action"] = torch.max(self.env.actions)

        #     extras["curriculum/distribution"] = {}
        #     for curriculum, category in zip(self.env.curricula, self.env.category_names):
        #         extras[f"curriculum/distribution"][f"weights_{category}"] = curriculum.weights
        #         extras[f"curriculum/distribution"][f"grid_{category}"] = curriculum.grid
        if self.env.cfg.env.send_timeouts:
            extras["time_outs"] = self.env.time_out_buf

        return extras
