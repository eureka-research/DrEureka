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
    
        # Calculate the velocity reward
        target_velocity = 2.0
        vel_error = torch.abs(env.base_lin_vel[:, 0] - target_velocity)
        velocity_reward = torch.exp(-vel_error / target_velocity)  # Using exponential to smooth the reward
    
        # Reward for maintaining torso height
        target_height = 0.34
        height_error = torch.abs(env.root_states[:, 2] - target_height)
        height_reward = torch.exp(-height_error / target_height)  # Using exponential to smooth the reward
    
        # Reward for maintaining orientation perpendicular to gravity
        orientation_reward = -torch.norm(env.projected_gravity[:, :2], dim=1)
    
        # Penalty for high action rates
        action_rate_penalty = torch.sum(torch.abs(env.actions - env.last_actions), dim=1)
    
        # Penalty for DOF limit violations
        dof_pos_penalty = torch.sum(torch.abs(env.dof_pos - env.default_dof_pos) > (env.dof_pos_limits[:, 1] - env.dof_pos_limits[:, 0]) * 0.5, dim=1).float()
    
        # Penalty for high torques
        torque_penalty = torch.sum(env.torques ** 2, dim=1)
    
        # Combining all reward terms with adjusted weights
        reward = (
            1.5 * velocity_reward
            + 0.5 * height_reward
            + 0.5 * orientation_reward
            - 0.005 * action_rate_penalty
            - 0.05 * dof_pos_penalty
            - 0.0005 * torque_penalty
        )
    
        # Normalizing reward terms
        reward_components = {
            "velocity_reward": velocity_reward,
            "height_reward": height_reward,
            "orientation_reward": orientation_reward,
            "action_rate_penalty": action_rate_penalty,
            "dof_pos_penalty": dof_pos_penalty,
            "torque_penalty": torque_penalty
        }
    
        return reward, reward_components
    
    # Success criteria as forward velocity
    def compute_success(self):
        target_velocity = 2.0
        lin_vel_error = torch.square(target_velocity - self.env.root_states[:, 7])
        return torch.exp(-lin_vel_error / 0.25)

