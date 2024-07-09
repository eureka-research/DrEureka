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
        
        # Calculate the forward velocity reward
        target_velocity = 2.0  # Target velocity in the x direction (m/s)
        vel_error = torch.abs(env.base_lin_vel[:, 0] - target_velocity)
        forward_vel_reward = 1.0 - vel_error / target_velocity
        forward_vel_reward = torch.clamp(forward_vel_reward, 0.0, 1.0)
        
        # Calculate the stability in height (z position)
        target_z_position = 0.34  # Target height in z direction (m)
        height_error = torch.abs(env.root_states[:, 2] - target_z_position)
        z_stability_reward = 1.0 - height_error
        z_stability_reward = torch.clamp(z_stability_reward, 0.0, 1.0)
        
        # Calculate the orientation stability (orientation should be perpendicular to gravity)
        projected_gravity = env.projected_gravity
        upright_reward = 1.0 - torch.norm(projected_gravity - env.gravity_vec, dim=1)
        upright_reward = torch.clamp(upright_reward, 0.0, 1.0)
        
        # Reward for minimal action rate to encourage smooth leg movements
        action_rate_cost = torch.sum(torch.abs(env.actions - env.last_actions), dim=-1)
        action_smoothness_reward = 1.0 - action_rate_cost
        action_smoothness_reward = torch.clamp(action_smoothness_reward, 0.0, 1.0)
        
        # Penalty for hitting the DOF limits
        dof_limits_penalty = (env.dof_pos < env.dof_pos_limits[:, 0]).float() + (env.dof_pos > env.dof_pos_limits[:, 1]).float()
        dof_limits_penalty = torch.sum(dof_limits_penalty, dim=-1)
        dof_limits_penalty = torch.clamp(dof_limits_penalty, 0.0, 1.0)
        smooth_leg_movement_reward = 1.0 - dof_limits_penalty
        
        # Total reward
        reward = 0.4 * forward_vel_reward + 0.2 * z_stability_reward + 0.2 * upright_reward + 0.1 * action_smoothness_reward + 0.1 * smooth_leg_movement_reward
        
        # Reward components for analysis
        reward_components = {
            'forward_vel_reward': forward_vel_reward,
            'z_stability_reward': z_stability_reward,
            'upright_reward': upright_reward,
            'action_smoothness_reward': action_smoothness_reward,
            'smooth_leg_movement_reward': smooth_leg_movement_reward
        }
        
        return reward, reward_components
    
    # Success criteria as forward velocity
    def compute_success(self):
        target_velocity = 2.0
        lin_vel_error = torch.square(target_velocity - self.env.root_states[:, 7])
        return torch.exp(-lin_vel_error / 0.25)

