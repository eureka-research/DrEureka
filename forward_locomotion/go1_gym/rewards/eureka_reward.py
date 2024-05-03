import torch
import numpy as np
from forward_locomotion.go1_gym.utils.math_utils import quat_apply_yaw, wrap_to_pi, get_scale_shift
from isaacgym.torch_utils import *

class EurekaReward():
    def __init__(self, env):
        self.env = env

    def load_env(self, env):
        self.env = env

    def compute_reward(self, using_curriculum=False):
        env = self.env  # Do not skip this line. Afterwards, use env.{parameter_name} to access parameters of the environment.
    
        # Ideal forward velocity in the x direction
        # target_velocity_x = 2.0
        target_velocity_x = self.env.cfg.rewards.target_velocity
        # Ideal height of the robot's torso
        target_height_z = 0.34
    
        # Compute the velocity reward component
        current_velocity_x = env.root_states[:, 7]  # Linear velocity in x from the root_states tensor
        velocity_error = torch.abs(current_velocity_x - target_velocity_x)
        velocity_reward = torch.exp(-velocity_error)
    
        # Compute the height reward component
        current_height = env.root_states[:, 2]  # Position in z from the root_states tensor
        height_error = torch.abs(current_height - target_height_z)
        height_reward = torch.exp(-5.0 * height_error)  # More weight to maintain height
    
        # Compute the orientation reward component
        # Ideal orientation is perpendicular to gravity, i.e., the projected gravity vector should be [0, 0, -1] in the robot's frame
        ideal_projected_gravity = torch.tensor([0., 0., -1.], device=env.device).repeat((env.num_envs, 1))
        orientation_error = torch.norm(env.projected_gravity - ideal_projected_gravity, dim=1)
        orientation_reward = torch.exp(-5.0 * orientation_error)  # More weight to maintain orientation
    
        # Legs movement within DOF limits reward component
        dof_limit_violations = torch.any(
            (env.dof_pos < env.dof_pos_limits[:, 0]) | (env.dof_pos > env.dof_pos_limits[:, 1]),
            dim=-1)
        dof_limit_violations_reward = 1.0 - dof_limit_violations.float()  # Penalize if any DOF limit is violated
    
        # Smoothness reward component (penalize the change in actions to encourage smooth movements)
        action_difference = torch.norm(env.actions - env.last_actions, dim=1)
        smoothness_reward = torch.exp(-0.1 * action_difference)
    
        # Combine reward components
        total_reward = velocity_reward * height_reward * orientation_reward * dof_limit_violations_reward * smoothness_reward
    
        # Debug information
        reward_components = {"velocity_reward": velocity_reward,
                             "height_reward": height_reward,
                             "orientation_reward": orientation_reward,
                             "dof_limit_violations_reward": dof_limit_violations_reward,
                             "smoothness_reward": smoothness_reward}

        if using_curriculum:
            # Additional terms, only used when training with curriculum
            def _reward_tracking_lin_vel(env):
                # Tracking of linear velocity commands (xy axes)
                if env.cfg.commands.global_reference:
                    lin_vel_error = torch.sum(torch.square(env.commands[:, :2] - env.root_states[:, 7:9]), dim=1)
                else:
                    lin_vel_error = torch.sum(torch.square(env.commands[:, :2] - env.base_lin_vel[:, :2]), dim=1)
                return torch.exp(-lin_vel_error / env.cfg.rewards.tracking_sigma)
            def _reward_tracking_ang_vel(env):
                # Tracking of angular velocity commands (yaw) 
                ang_vel_error = torch.square(env.commands[:, 2] - env.base_ang_vel[:, 2])
                return torch.exp(-ang_vel_error / env.cfg.rewards.tracking_sigma_yaw)
            reward_components["tracking_lin_vel"] = _reward_tracking_lin_vel(env)
            reward_components["tracking_ang_vel"] = _reward_tracking_ang_vel(env)
                             
        return total_reward, reward_components
    
    # Success criteria as forward velocity
    def compute_success(self):
        target_velocity = 2.0
        lin_vel_error = torch.square(target_velocity - self.env.root_states[:, 7])
        return torch.exp(-lin_vel_error / 0.25)