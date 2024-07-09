    def compute_reward(self):
        env = self.env  # Do not skip this line. Afterwards, use env.{parameter_name} to access parameters of the environment.
    
        # Constants
        desired_velocity = 2.0
        desired_height = 0.34
        upright_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device)  # Assumes that upright orientation quaternion is [1, 0, 0, 0]
    
        # Current velocity in global frame
        current_velocity = env.root_states[:, 7:10]  # Shape (num_envs, 3)
    
        # Reward for running at the desired velocity (along positive x direction)
        vel_forward_reward = torch.exp(-0.5 * torch.square(current_velocity[:, 0] - desired_velocity))
    
        # Reward for maintaining the desired height of the torso
        height_reward = torch.exp(-10.0 * torch.square(env.root_states[:, 2] - desired_height))
    
        # Reward for the robot being upright (aligning base_quat with upright_quat)
        rotation_diff = torch.sum(env.base_quat * upright_quat, dim=-1)
        upright_reward = torch.exp(-20.0 * (1.0 - torch.abs(rotation_diff)))
    
        # Penalty for action rate (encourages smooth movements)
        action_rate_penalty = torch.exp(-torch.square(env.actions - env.last_actions)).sum(dim=-1)
    
        # Penalty for exceeding joint limits
        dof_pos_penalty = torch.reduce_sum(torch.square(env.dof_pos - env.default_dof_pos), dim=1)
        dof_vel_penalty = torch.reduce_sum(torch.square(env.dof_vel), dim=1)
    
        # Weighted sum of rewards
        reward = 2.0 * vel_forward_reward + 1.0 * height_reward + 1.0 * upright_reward - 0.1 * action_rate_penalty - 0.01 * dof_pos_penalty - 0.01 * dof_vel_penalty
    
        # Collect individual reward components for debugging purposes
        reward_components = {
            'vel_forward_reward': vel_forward_reward,
            'height_reward': height_reward,
            'upright_reward': upright_reward,
            'action_rate_penalty': action_rate_penalty,
            'dof_pos_penalty': dof_pos_penalty,
            'dof_vel_penalty': dof_vel_penalty
        }
    
        return reward, reward_components
