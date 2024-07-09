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
    
        # Reward for running forward at 2.0 m/s (positive x direction)
        target_velocity = 2.0
        forward_velocity_error = torch.abs(env.base_lin_vel[:, 0] - target_velocity)
        forward_reward = torch.exp(-2 * forward_velocity_error)
        
        # Penalty for deviation from the target z position (0.34 m)
        target_z_position = 0.34
        z_position_error = torch.abs(env.root_states[:, 2] - target_z_position)
        z_position_penalty = torch.exp(-10 * z_position_error)
        
        # Penalty for deviation from the perpendicular orientation to gravity
        orientation_penalty = torch.exp(-10 * torch.abs(env.projected_gravity[:, 2] - 1.0))
        
        # Reward for minimal action rate (difference between consecutive actions)
        action_rate_penalty = torch.sum(torch.abs(env.actions - env.last_actions), dim=-1)
        
        # Penalty for avoiding DOF position limits
        dof_pos_limits_min = env.dof_pos_limits[:, 0]
        dof_pos_limits_max = env.dof_pos_limits[:, 1]
        dof_pos = env.dof_pos
        dof_pos_violation_penalty = torch.sum(
            torch.max(torch.zeros_like(dof_pos), dof_pos - dof_pos_limits_max) +
            torch.max(torch.zeros_like(dof_pos), dof_pos_limits_min - dof_pos), dim=-1
        )
        
        # Penalty for smooth movements (minimizing DOF velocity)
        dof_velocity_penalty = torch.sum(torch.abs(env.dof_vel), dim=-1)
    
        # Final reward, sum of all components normalized and weighted as required
        total_reward = (2.0 * forward_reward + 
                        1.0 * z_position_penalty +
                        1.0 * orientation_penalty - 
                        0.1 * action_rate_penalty - 
                        0.5 * dof_pos_violation_penalty - 
                        0.5 * dof_velocity_penalty)
        
        reward_components = {
            'forward_reward': forward_reward,
            'z_position_penalty': z_position_penalty,
            'orientation_penalty': orientation_penalty,
            'action_rate_penalty': action_rate_penalty,
            'dof_pos_violation_penalty': dof_pos_violation_penalty,
            'dof_velocity_penalty': dof_velocity_penalty
        }
    
        return total_reward, reward_components
    
    # Success criteria as forward velocity
    def compute_success(self):
        target_velocity = 2.0
        lin_vel_error = torch.square(target_velocity - self.env.root_states[:, 7])
        return torch.exp(-lin_vel_error / 0.25)

