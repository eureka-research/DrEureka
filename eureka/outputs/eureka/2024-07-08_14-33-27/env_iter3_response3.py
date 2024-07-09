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
    
        # Calculate the velocity reward based on how close the robot's base linear velocity in the x direction is to 2.0 m/s
        target_velocity = 2.0
        vel_error = torch.abs(env.base_lin_vel[:, 0] - target_velocity)
        velocity_reward = 2.0 * (1.0 - vel_error / target_velocity)
    
        # Reward for maintaining torso height near 0.34 meters
        target_height = 0.34
        height_error = torch.abs(env.root_states[:, 2] - target_height)
        height_reward = 1.5 * (1.0 - height_error / target_height)
    
        # Reward for maintaining orientation perpendicular to gravity
        orientation_error = torch.norm(env.projected_gravity[:, :2], dim=1)  # Should be close to zero
        orientation_reward = -0.1 * orientation_error
    
        # Penalty for high action rates (difference between consecutive actions)
        action_rate_penalty = 0.00002 * torch.sum(torch.abs(env.actions - env.last_actions), dim=1)
    
        # Penalty for DOF limit violations
        dof_pos_error = torch.abs(env.dof_pos - env.default_dof_pos)
        dof_pos_penalty = 0.02 * torch.sum((dof_pos_error > 
                                          (env.dof_pos_limits[:, 1] - env.dof_pos_limits[:, 0]) * 0.5).float(), dim=1)
    
        # Penalty for high torques to encourage energy efficiency
        torque_penalty = 0.000002 * torch.sum(env.torques ** 2, dim=1)
        
        # Combine all reward terms
        reward = (
            velocity_reward
            + height_reward
            + orientation_reward
            - action_rate_penalty
            - dof_pos_penalty
            - torque_penalty
        )
    
        # Returning individual reward components for logging/debugging purposes
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

