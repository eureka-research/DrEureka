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
    
        # Extract relevant parameters and metrics
        base_lin_vel = env.base_lin_vel  # Linear velocity of the torso
        base_pos_z = env.root_states[:, 2]  # z position of the torso
        projected_gravity = env.projected_gravity  # Orientation of the torso
        dof_vel = env.dof_vel  # Velocity of each DOF
        dof_pos = env.dof_pos  # Position of each DOF
        last_dof_vel = env.last_dof_vel  # Velocity of each DOF at last time step
        actions = env.actions  # Actions taken by the agent
        last_actions = env.last_actions  # Actions taken by the agent at last time step
        dof_pos_limits = env.dof_pos_limits  # DOF position limits
    
        # Reward components
        base_vel_reward = -torch.abs(base_lin_vel[:, 0] - 2.0)  # Encourage a forward velocity of 2.0 m/s
        torso_height_reward = -torch.abs(base_pos_z - 0.34)  # Encourage torso z position to be near 0.34
        orientation_reward = -torch.norm(projected_gravity[:, [0, 1]], dim=1)  # Minimize deviation from perpendicular to gravity
        smoothness_reward = -torch.mean((dof_vel - last_dof_vel) ** 2, dim=1)  # Encourage smooth leg movements
        action_rate_penalty = -torch.mean((actions - last_actions) ** 2, dim=1)  # Penalize high action rates
        dof_limits_penalty = -torch.sum(torch.maximum(dof_pos - dof_pos_limits[:, 1], torch.tensor(0.0, device=self.device)) + torch.maximum(dof_pos_limits[:, 0] - dof_pos, torch.tensor(0.0, device=self.device)), dim=1)  # Penalize exceeding DOF limits
    
        # Total reward
        reward = base_vel_reward + torso_height_reward + orientation_reward + smoothness_reward + action_rate_penalty + dof_limits_penalty
    
        # Reward components dictionary
        reward_components = {
            "base_vel_reward": base_vel_reward,
            "torso_height_reward": torso_height_reward,
            "orientation_reward": orientation_reward,
            "smoothness_reward": smoothness_reward,
            "action_rate_penalty": action_rate_penalty,
            "dof_limits_penalty": dof_limits_penalty
        }
    
        return reward, reward_components
    
    # Success criteria as forward velocity
    def compute_success(self):
        target_velocity = 2.0
        lin_vel_error = torch.square(target_velocity - self.env.root_states[:, 7])
        return torch.exp(-lin_vel_error / 0.25)

