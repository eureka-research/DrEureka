import torch
import numpy as np
from globe_walking.go1_gym.utils.math_utils import quat_apply_yaw, wrap_to_pi, get_scale_shift
from isaacgym.torch_utils import *

class EurekaReward():
    def __init__(self, env):
        self.env = env

    def load_env(self, env):
        self.env = env
    
    def _reward_height(self):
        env = self.env
        height_threshold = 2.0 * env.ball_radius
        height_temperature = 7.0  # Fine-tuned temperature parameter
        height_exp = torch.exp((env.base_pos[:, 2] - height_threshold) / height_temperature)
        height_reward = torch.where(env.base_pos[:, 2] >= height_threshold, height_exp, torch.zeros_like(env.base_pos[:, 2]))
        return 1.5 * height_reward  # Updated scaling
    
    def _reward_balance(self):
        env = self.env
        balance_temperature = 5.0  # Fine-tuned temperature parameter
        # ball_top = env.object_pos_world_frame + torch.tensor([0.0, 0.0, env.ball_radius], device=env.device).unsqueeze(0)
        ball_top = env.object_pos_world_frame.clone()
        ball_top[:, 2] += env.ball_radius

        feet_dist_to_ball_top = torch.norm(env.foot_positions - ball_top.unsqueeze(1), dim=-1)
        balance_exp = torch.exp(-feet_dist_to_ball_top / balance_temperature)
        balance_reward = torch.mean(balance_exp, dim=-1)
        return 2.0 * balance_reward  # Updated scaling
    
    def _reward_smooth_actions(self):
        env = self.env
        action_diff = env.actions - env.last_actions
        smooth_actions_reward = -torch.mean(torch.abs(action_diff), dim=-1)
        return 1.0 * smooth_actions_reward  # Increase scale of smooth_actions_reward
    
    def _reward_penalize_large_actions(self):
        env = self.env
        large_action_penalty = -torch.mean(torch.abs(env.actions), dim=-1)
        return 0.3 * large_action_penalty  # Increase scaling for penalize_large_actions

    # Success criteria as episode length
    def compute_success(self):
        return torch.ones_like(self.env.base_pos[:, 2])
