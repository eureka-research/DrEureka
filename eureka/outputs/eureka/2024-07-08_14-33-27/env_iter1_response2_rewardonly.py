    def compute_reward(self):
        env = self.env  # Do not skip this line. Afterwards, use env.{parameter_name} to access parameters of the environment.
    
        # Calculate the velocity reward
        target_velocity = 2.0
        vel_error = torch.abs(env.base_lin_vel[:, 0] - target_velocity)
        velocity_reward = torch.exp(-vel_error / target_velocity)  # Using exponential to smooth the reward
    
        # Reward for maintaining torso height
        target_height = 0.34
        height_error = torch.abs(env.root_states[:, 2] - target_height)
        height_reward = torch.exp(-height_error / target_height)  # Using exponential to smooth the reward
    
        # Reward for maintaining orientation perpendicular to gravity
        orientation_reward = -torch.norm(env.projected_gravity[:, :2], dim=1)
    
        # Penalty for high action rates
        action_rate_penalty = torch.sum(torch.abs(env.actions - env.last_actions), dim=1)
    
        # Penalty for DOF limit violations
        dof_pos_penalty = torch.sum(torch.abs(env.dof_pos - env.default_dof_pos) > (env.dof_pos_limits[:, 1] - env.dof_pos_limits[:, 0]) * 0.5, dim=1).float()
    
        # Penalty for high torques
        torque_penalty = torch.sum(env.torques ** 2, dim=1)
    
        # Combining all reward terms with adjusted weights
        reward = (
            1.5 * velocity_reward
            + 0.5 * height_reward
            + 0.5 * orientation_reward
            - 0.005 * action_rate_penalty
            - 0.05 * dof_pos_penalty
            - 0.0005 * torque_penalty
        )
    
        # Normalizing reward terms
        reward_components = {
            "velocity_reward": velocity_reward,
            "height_reward": height_reward,
            "orientation_reward": orientation_reward,
            "action_rate_penalty": action_rate_penalty,
            "dof_pos_penalty": dof_pos_penalty,
            "torque_penalty": torque_penalty
        }
    
        return reward, reward_components
