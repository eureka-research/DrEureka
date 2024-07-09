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
    
        # Velocity reward: increase emphasis for achieving target velocity (2.0 m/s)
        target_velocity = 2.0
        vel_error = torch.abs(env.base_lin_vel[:, 0] - target_velocity)
        velocity_reward = 2.5 * (1.0 - vel_error / target_velocity)
    
        # Height reward: Ensure stability in maintaining target height (0.34 m)
        target_height = 0.34
        height_error = torch.abs(env.root_states[:, 2] - target_height)
        height_reward = 1.0 * (1.0 - height_error / target_height)
    
        # Orientation reward: Refine and rescale
        orientation_error = torch.norm(env.projected_gravity[:, :2], dim=1)
        orientation_reward = 0.5 * (1.0 - orientation_error / torch.norm(env.gravity_vec[:, :2]))
    
        # Action rate penalty: Significantly reduce
        action_rate_penalty = 0.00005 * torch.sum(torch.abs(env.actions - env.last_actions), dim=1)
    
        # DOF position penalty: Significant reduction
        dof_pos_penalty = 0.01 * torch.sum((torch.abs(env.dof_pos - env.default_dof_pos) >
                                          (env.dof_pos_limits[:, 1] - env.dof_pos_limits[:, 0]) * 0.5).float(), dim=1)
    
        # Torque penalty: Reduce further
        torque_penalty = 0.000005 * torch.sum(env.torques ** 2, dim=1)
    
        # Total reward combination
        reward = (
            velocity_reward  # 2.5x scaling (Ultermore task emphasizing)
            + height_reward  # 1.0x scaling
            + orientation_reward  # 0.5x scaling
            - action_rate_penalty  # 0.00005x scaling
            - dof_pos_penalty  # 0.01x coding
            - torque_penalty  # 0.000005x scaling
        )
    
        # Individual components dictionary (for logging/troubleshooting)
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

