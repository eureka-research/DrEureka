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
        import torch
    
        # Desired forward velocity
        desired_velocity = 2.0
    
        # Calculate components of the reward
        # 1. Forward velocity reward
        forward_velocity = env.root_states[:, 7]  # X-direction linear velocity
        forward_velocity_reward = -torch.abs(forward_velocity - desired_velocity)
    
        # 2. Stability reward (Height and orientation control)
        target_z_pos = 0.34
        base_height = env.root_states[:, 2]  # Z position of the torso
        height_reward = -torch.abs(base_height - target_z_pos)
    
        orientation_reward = -torch.abs(env.projected_gravity[:, 2] - 1.0)  # Should be close to 1 if perpendicular to gravity
    
        # 3. Penalize high action rate for smoothness
        action_smoothness_reward = -torch.sum(torch.abs(env.actions - env.last_actions), dim=1)
    
        # 4. Penalize DOF limit violations for joint position and velocity
        dof_pos_penalty = torch.sum(
            (env.dof_pos - env.dof_pos_limits[:, 0].unsqueeze(0)).clamp(min=0)
            + (env.dof_pos - env.dof_pos_limits[:, 1].unsqueeze(0)).clamp(max=0),
            dim=1
        )
    
        dof_vel_limit = env.dof_vel_limits.unsqueeze(0)
        dof_vel_penalty = torch.sum(
            torch.abs(env.dof_vel) - dof_vel_limit * (torch.abs(env.dof_vel) > dof_vel_limit).float(),
            dim=1
        )
    
        # Combine all the rewards
        total_reward = (
            forward_velocity_reward
            + 0.5 * height_reward
            + 0.5 * orientation_reward
            + 0.1 * action_smoothness_reward
            - 0.1 * (dof_pos_penalty + dof_vel_penalty)
        )
    
        # Create a dictionary of each individual reward component
        reward_dict = {
            'forward_velocity_reward': forward_velocity_reward,
            'height_reward': height_reward,
            'orientation_reward': orientation_reward,
            'action_smoothness_reward': action_smoothness_reward,
            'dof_pos_penalty': dof_pos_penalty,
            'dof_vel_penalty': dof_vel_penalty,
        }
    
        return total_reward, reward_dict
    
    # Success criteria as forward velocity
    def compute_success(self):
        target_velocity = 2.0
        lin_vel_error = torch.square(target_velocity - self.env.root_states[:, 7])
        return torch.exp(-lin_vel_error / 0.25)

