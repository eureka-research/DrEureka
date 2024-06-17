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
    
        # Reward for target forward velocity
        forward_speed_error = torch.abs(env.base_lin_vel[:, 0] - 2.0)
        reward_forward_speed = torch.exp(-3 * forward_speed_error)  # Increased influence for better precision
    
        # Reward for maintaining torso height close to 0.34m
        height_error = torch.abs(env.root_states[:, 2] - 0.34)
        reward_height = torch.exp(-8 * height_error)  # Reduced weight for balance
    
        # Reward for maintaining stable orientation
        orientation_error = torch.norm(env.projected_gravity - env.gravity_vec, dim=1)
        reward_orientation = torch.exp(-8 * orientation_error)  # Reduced weight for balance
    
        # Penalty for excessive actions (action smoothness) - Rewritten
        action_rate = torch.square(env.actions - env.last_actions).sum(dim=1)
        reward_action_smoothness = torch.exp(-10 * action_rate)  # Increased scaling factor for relevance
    
        # Penalty for DOF limits violation - Softened further
        dof_position_penalty = -torch.sum(torch.abs(env.dof_pos - env.default_dof_pos), dim=1) * 0.02
    
        # Total reward
        total_reward = (2.0 * reward_forward_speed 
                        + 0.2 * reward_height 
                        + 0.2 * reward_orientation 
                        + 0.5 * reward_action_smoothness 
                        + dof_position_penalty)
    
        # Dictionary of each individual reward component
        reward_components = {
            'reward_forward_speed': reward_forward_speed,
            'reward_height': reward_height,
            'reward_orientation': reward_orientation,
            'reward_action_smoothness': reward_action_smoothness,
            'dof_position_penalty': dof_position_penalty
        }
    
        return total_reward, reward_components
    
    # Success criteria as forward velocity
    def compute_success(self):
        target_velocity = 2.0
        lin_vel_error = torch.square(target_velocity - self.env.root_states[:, 7])
        return torch.exp(-lin_vel_error / 0.25)


