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
        
        # Forward velocity reward
        forward_vel_reward = torch.exp(-torch.abs(env.base_lin_vel[:, 0] - 2.0))
    
        # Height reward (assuming z position is the 2nd element of the position array)
        height_reward = torch.exp(-torch.abs(env.root_states[:, 2] - 0.34)) * 0.5
        
        # Orientation reward (projected gravity should align with the robot's z-axis in local frame)
        orientation_reward = torch.exp(-torch.norm(env.projected_gravity - env.gravity_vec, dim=-1)) * 0.5
        
        # Smooth actions reward (penalizing differences in actions to encourage smoothness)
        action_smoothness_reward = torch.exp(-torch.norm(env.actions - env.last_actions, dim=-1))
        
        # Avoid DOF limits reward (punishing if DOF positions are near limits)
        dof_limit_penalty = torch.sum(torch.square(env.dof_pos - (env.dof_pos_limits[:, 0] + env.dof_pos_limits[:, 1]) / 2)
                                      / ((env.dof_pos_limits[:, 1] - env.dof_pos_limits[:, 0]) / 2)**2, dim=-1)
        dof_limit_reward = torch.exp(-dof_limit_penalty) * 0.3
        
        # Episode length penalty to discourage early terminations
        episode_termination_penalty = torch.ones_like(forward_vel_reward) * -0.1
        
        # Weighting of different rewards
        reward = (3.0 * forward_vel_reward + height_reward + orientation_reward + 
                  action_smoothness_reward + dof_limit_reward + episode_termination_penalty)
    
        # Normalize the reward components to lie between 0 and 1 before summing
        total_reward = reward.mean(dim=0)
        
        return total_reward, {
            "forward_vel_reward": forward_vel_reward,
            "height_reward": height_reward,
            "orientation_reward": orientation_reward,
            "action_smoothness_reward": action_smoothness_reward,
            "dof_limit_reward": dof_limit_reward,
            "episode_termination_penalty": episode_termination_penalty,
        }
    
    # Success criteria as forward velocity
    def compute_success(self):
        target_velocity = 2.0
        lin_vel_error = torch.square(target_velocity - self.env.root_states[:, 7])
        return torch.exp(-lin_vel_error / 0.25)

