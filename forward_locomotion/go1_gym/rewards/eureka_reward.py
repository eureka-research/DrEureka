import torch
import numpy as np
from forward_locomotion.go1_gym.utils.math_utils import quat_apply_yaw, wrap_to_pi, get_scale_shift
from isaacgym.torch_utils import *

class EurekaReward():
    def __init__(self, env):
        self.env = env

    def load_env(self, env):
        self.env = env

    def compute_reward(self):
        env = self.env  # Do not skip this line. Afterwards, use env.{parameter_name} to access parameters of the environment.
        
        # Desired forward velocity
        desired_velocity = 2.0
    
        # Reward for maintaining forward velocity
        forward_velocity_error = torch.abs(env.root_states[:, 7] - desired_velocity)
        forward_velocity_reward = torch.exp(-forward_velocity_error)
    
        # Reward for maintaining torso height near 0.34
        torso_height_target = 0.34
        height_error = torch.abs(env.root_states[:, 2] - torso_height_target)
        height_reward = torch.exp(-height_error)
    
        # Reward for maintaining orientation perpendicular to gravity
        orientation_reward = torch.exp(-torch.abs(env.projected_gravity[:, 2] - 1.0))
    
        # Reward for smoothness in actions
        action_rate = torch.sum(torch.abs(env.actions - env.last_actions), dim=1)
        action_smoothness_reward = torch.exp(-action_rate)
    
        # Reward for avoiding DOF limits
        dof_pos_limits_lower = env.dof_pos_limits[:, 0].unsqueeze(0).expand_as(env.dof_pos)
        dof_pos_limits_upper = env.dof_pos_limits[:, 1].unsqueeze(0).expand_as(env.dof_pos)
        dof_pos_penalty = torch.sum(torch.clamp(env.dof_pos - dof_pos_limits_upper, min=0) ** 2 + torch.clamp(dof_pos_limits_lower - env.dof_pos, min=0) ** 2, dim=1)
        dof_pos_limit_reward = torch.exp(-dof_pos_penalty)
    
        # Total reward
        reward = forward_velocity_reward + height_reward + orientation_reward + action_smoothness_reward + dof_pos_limit_reward
    
        # Dictionary of reward components
        reward_components = {
            "forward_velocity_reward": forward_velocity_reward,
            "height_reward": height_reward,
            "orientation_reward": orientation_reward,
            "action_smoothness_reward": action_smoothness_reward,
            "dof_pos_limit_reward": dof_pos_limit_reward
        }
    
        return reward, reward_components
    
    # Success criteria as forward velocity
    def compute_success(self):
        target_velocity = 2.0
        lin_vel_error = torch.square(target_velocity - self.env.root_states[:, 7])
        return torch.exp(-lin_vel_error / 0.25)

