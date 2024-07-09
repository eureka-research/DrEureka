    def compute_reward(self):
        env = self.env  # Do not skip this line. Afterwards, use env.{parameter_name} to access parameters of the environment.
        
        # Calculate the velocity reward based on how close the robot's base linear velocity in the x direction is to 2.0 m/s
        target_velocity = 2.0
        vel_error = torch.abs(env.base_lin_vel[:, 0] - target_velocity)
        velocity_reward = 1.0 - vel_error / target_velocity
    
        # Reward for maintaining torso height near 0.34 meters
        target_height = 0.34
        height_error = torch.abs(env.root_states[:, 2] - target_height)
        height_reward = 1.0 - height_error / target_height
    
        # Reward for maintaining orientation perpendicular to gravity
        orientation_reward = -torch.norm(env.projected_gravity[:, :2], dim=1)  # Should be close to zero
    
        # Penalty for high action rates (difference between consecutive actions)
        action_rate_penalty = torch.sum(torch.abs(env.actions - env.last_actions), dim=1)
    
        # Penalty for DOF limit violations
        dof_pos_penalty = torch.sum(torch.abs(env.dof_pos - env.default_dof_pos) > (env.dof_pos_limits[:, 1] - env.dof_pos_limits[:, 0]) * 0.5, dim=1).float()
    
        # Penalty for high torques to encourage energy efficiency
        torque_penalty = torch.sum(env.torques ** 2, dim=1)
    
        # Scaling factors
        velocity_reward_scale = 2.0
        height_reward_scale = 0.5
        orientation_reward_scale = 1.0
        action_rate_penalty_scale = 0.001  # Reduced
        dof_pos_penalty_scale = 0.1
        torque_penalty_scale = 0.0001  # Reduced
        
        # Combine all reward terms
        reward = (
            velocity_reward_scale * velocity_reward
            + height_reward_scale * height_reward
            + orientation_reward_scale * orientation_reward
            - action_rate_penalty_scale * action_rate_penalty
            - dof_pos_penalty_scale * dof_pos_penalty
            - torque_penalty_scale * torque_penalty
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
