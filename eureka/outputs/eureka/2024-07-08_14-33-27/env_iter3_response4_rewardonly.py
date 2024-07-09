    def compute_reward(self):
        env = self.env
    
        # Target values for velocity, height, and penalties
        target_velocity = 2.0
        target_height = 0.34
    
        # Velocity reward
        vel_error = torch.abs(env.base_lin_vel[:, 0] - target_velocity)
        velocity_reward = 1.0 * (1.0 - vel_error / target_velocity)
    
        # Height reward 
        height_error = torch.abs(env.root_states[:, 2] - target_height)
        height_reward = 0.5 * (1.0 - height_error / target_height)
    
        # Orientation reward 
        orientation_error = torch.norm(env.projected_gravity[:, :2], dim=1)
        orientation_reward = -0.5 * orientation_error
    
        # Action rate penalty
        action_rate_penalty = 0.00001 * torch.sum(torch.abs(env.actions - env.last_actions), dim=1)
    
        # DOF position penalty
        dof_pos_penalty = 0.01 * torch.sum((torch.abs(env.dof_pos - env.default_dof_pos) > 
                                          (env.dof_pos_limits[:, 1] - env.dof_pos_limits[:, 0]) * 0.5).float(), dim=1)
    
        # Torque penalty
        torque_penalty = 0.000001 * torch.sum(env.torques ** 2, dim=1)
    
        # Combined reward
        reward = (
            1.5 * velocity_reward
            + height_reward
            + orientation_reward
            - action_rate_penalty
            - dof_pos_penalty
            - torque_penalty
        )
    
        # Reward components dictionary
        reward_components = {
            "velocity_reward": velocity_reward,
            "height_reward": height_reward,
            "orientation_reward": orientation_reward,
            "action_rate_penalty": action_rate_penalty,
            "dof_pos_penalty": dof_pos_penalty,
            "torque_penalty": torque_penalty
        }
    
        return reward, reward_components
