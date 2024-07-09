    def compute_reward(self):
        env = self.env  # Do not skip this line. Afterwards, use env.{parameter_name} to access parameters of the environment.
    
        # Target values
        target_velocity = 2.0
        target_height = 0.34
    
        # Calculate individual reward components
        vel_error = torch.abs(env.base_lin_vel[:, 0] - target_velocity)
        velocity_reward = 1.0 - vel_error / target_velocity
    
        height_error = torch.abs(env.root_states[:, 2] - target_height)
        height_reward = 1.0 - height_error / target_height
    
        orientation_reward = -torch.norm(env.projected_gravity[:, :2], dim=1)
    
        action_rate_penalty = torch.sum(torch.abs(env.actions - env.last_actions), dim=1)
    
        dof_pos_penalty = torch.sum(torch.abs(env.dof_pos - env.default_dof_pos) > 
                                     (env.dof_pos_limits[:, 1] - env.dof_pos_limits[:, 0]) * 0.5, dim=1).float()
    
        torque_penalty = torch.sum(env.torques ** 2, dim=1)
    
        # Scale the reward components to balance their influence
        velocity_reward *= 2.0
        height_reward *= 1.0
        orientation_reward *= 1.0
        action_rate_penalty *= 0.005
        dof_pos_penalty *= 0.1
        torque_penalty *= 0.0001
        
        # Combining all reward terms
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
