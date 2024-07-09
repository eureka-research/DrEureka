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
        # Desired torso height and orientation
        desired_z_position = 0.34
    
        # Compute the actual linear velocity in the global x direction
        actual_velocity = env.root_states[:, 7]
    
        # Compute deviation from the desired forward velocity
        velocity_error = torch.abs(actual_velocity - desired_velocity)
    
        # Reward for forward velocity accuracy (penalize deviation from desired_velocity)
        velocity_reward = -velocity_error
    
        # Compute the torso's z position
        torso_z_position = env.root_states[:, 2]
    
        # Compute deviation from the desired torso height
        height_error = torch.abs(torso_z_position - desired_z_position)
    
        # Reward for maintaining the desired torso height
        height_reward = -height_error
    
        # Penalize angular deviation from the upright orientation
        orientation_error = torch.norm(env.projected_gravity[:, :2], dim=-1)
        orientation_reward = -orientation_error
    
        # Penalize large actions (action smoothness)
        action_rate_penalty = torch.sum(torch.abs(env.actions - env.last_actions), dim=-1)
        action_rate_penalty = -action_rate_penalty
    
        # Penalize reaching the DOF limits.
        dof_pos_penalty = torch.sum((env.dof_pos.abs() > (env.dof_pos_limits[:, 1] - env.dof_pos_limits[:, 0])).float(), dim=-1)
        dof_pos_penalty = -dof_pos_penalty
    
        # Sum up the individual rewards to get the total reward
        total_reward = velocity_reward + height_reward + orientation_reward + action_rate_penalty + dof_pos_penalty
    
        return total_reward, {
            'velocity_reward': velocity_reward,
            'height_reward': height_reward,
            'orientation_reward': orientation_reward,
            'action_rate_penalty': action_rate_penalty,
            'dof_pos_penalty': dof_pos_penalty,
        }
    
    # Success criteria as forward velocity
    def compute_success(self):
        target_velocity = 2.0
        lin_vel_error = torch.square(target_velocity - self.env.root_states[:, 7])
        return torch.exp(-lin_vel_error / 0.25)

