    def compute_reward(self):
        env = self.env  # Do not skip this line. Afterwards, use env.{parameter_name} to access parameters of the environment.
        # Constants
        target_velocity = 2.0  # Target velocity in m/s
        target_torso_z = 0.34  # Target torso height in meters
        
        # Extract relevant parameters from environment
        base_lin_vel_x = env.base_lin_vel[:, 0]
        torso_z = env.root_states[:, 2]
        orientation_error = 1.0 - torch.abs(env.projected_gravity[:, 2])  # Orientation perpendicular to gravity
        dof_limits_exceeded = ((env.dof_pos < env.dof_pos_limits[:, 0]) | 
                               (env.dof_pos > env.dof_pos_limits[:, 1])).float()
        
        # Component rewards
        vel_reward = -torch.abs(base_lin_vel_x - target_velocity)
        height_reward = -torch.abs(torso_z - target_torso_z)
        orientation_reward = -orientation_error
        smoothness_reward = -torch.mean(torch.abs(env.dof_vel - env.last_dof_vel), dim=1)
        dof_limit_penalty = -torch.sum(dof_limits_exceeded, dim=1)
        min_action_rate_penalty = -torch.mean(torch.abs(env.actions - env.last_actions), dim=1)
        
        # Total reward
        reward = (0.5 * vel_reward + 
                  0.2 * height_reward + 
                  0.2 * orientation_reward + 
                  0.05 * smoothness_reward + 
                  0.05 * dof_limit_penalty + 
                  0.05 * min_action_rate_penalty)
        
        # Create reward components dictionary
        reward_components = {
            "velocity_reward": vel_reward,
            "height_reward": height_reward,
            "orientation_reward": orientation_reward,
            "smoothness_reward": smoothness_reward,
            "dof_limit_penalty": dof_limit_penalty,
            "min_action_rate_penalty": min_action_rate_penalty
        }
        
        return reward, reward_components
