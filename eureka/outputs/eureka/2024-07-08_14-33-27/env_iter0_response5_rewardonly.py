    def compute_reward(self):
        env = self.env  # Do not skip this line. Afterwards, use env.{parameter_name} to access parameters of the environment.
    
        # Desired velocity in positive x direction
        target_velocity = 2.0
    
        # Linear velocity in the base frame
        base_lin_vel = env.base_lin_vel[:, 0]  # Velocity in x direction
    
        # Position of the base (z position of torso)
        base_z_pos = env.root_states[:, 2]
    
        # Projected gravity in the base frame (orientation check)
        projected_gravity = env.projected_gravity[:, 2]
    
        # Action differences to penalize large action changes
        action_diff = env.actions - env.last_actions
    
        # DOF limit violation penalties
        dof_limits_lower = env.dof_pos_limits[:, 0]
        dof_limits_upper = env.dof_pos_limits[:, 1]
        dof_pos = env.dof_pos
    
        dof_limit_penalty = ((dof_pos < dof_limits_lower).float() + (dof_pos > dof_limits_upper).float()).sum(dim=1)
    
        # Calculate each individual reward component
        velocity_reward = -((base_lin_vel - target_velocity) ** 2)
        stability_reward = -((base_z_pos - 0.34) ** 2)
        orientation_reward = -(projected_gravity ** 2)
        action_smoothness_reward = -torch.sum(action_diff ** 2, dim=1)
        dof_smoothness_reward = torch.sum(-torch.abs(env.dof_vel - env.last_dof_vel), dim=1)
        dof_limit_reward = -dof_limit_penalty
    
        # Aggregating the rewards
        reward = (velocity_reward + stability_reward + orientation_reward + action_smoothness_reward +
                  dof_smoothness_reward + dof_limit_reward)
    
        reward_components = {
            'velocity_reward': velocity_reward,
            'stability_reward': stability_reward,
            'orientation_reward': orientation_reward,
            'action_smoothness_reward': action_smoothness_reward,
            'dof_smoothness_reward': dof_smoothness_reward,
            'dof_limit_reward': dof_limit_reward
        }
    
        return reward, reward_components
