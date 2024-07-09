    def compute_reward(self):
        env = self.env  # Do not skip this line. Afterwards, use env.{parameter_name} to access parameters of the environment.
    
        # Calculate individual reward components
        # 1. Velocity Reward: Encourages forward movement at 2.0 m/s in the positive x direction
        velocity_error = torch.abs(env.base_lin_vel[:, 0] - 2.0)
        velocity_reward = 1.0 - velocity_error
    
        # 2. Stability Rewards: Ensures steady and stable policy
        # a. Maintain torso height around 0.34 meters
        height_error = torch.abs(env.root_states[:, 2] - 0.34)
        height_reward = 1.0 - height_error
    
        # b. Maintain proper orientation (i.e., perpendicular to gravity)
        orientation_error = torch.norm(env.projected_gravity - env.gravity_vec, dim=1)
        orientation_reward = 1.0 - orientation_error
    
        # 3. Smoothness of leg movements
        smooth_movement_reward = -torch.norm(env.dof_vel, dim=1)
    
        # 4. Minimize action rate
        action_rate_penalty = -torch.norm(env.actions - env.last_actions, dim=1)
    
        # 5. Avoiding DOF limits
        pos_limits_lower_error = torch.min(env.dof_pos - env.dof_pos_limits[:, 0], torch.zeros_like(env.dof_pos))
        pos_limits_upper_error = torch.min(env.dof_pos_limits[:, 1] - env.dof_pos, torch.zeros_like(env.dof_pos))
        dof_limit_penalty = pos_limits_lower_error.sum(dim=1) + pos_limits_upper_error.sum(dim=1)
    
        # Weigh the components to get the total reward
        total_reward = velocity_reward * 1.0 + \
                       height_reward * 0.5 + \
                       orientation_reward * 0.5 + \
                       smooth_movement_reward * 0.2 + \
                       action_rate_penalty * 0.2 + \
                       dof_limit_penalty * 0.1
    
        # Ensure rewards are non-negative
        total_reward = torch.clamp(total_reward, min=0.0)
        
        # Return total reward and individual component rewards
        reward_components = {
            "velocity_reward": velocity_reward,
            "height_reward": height_reward,
            "orientation_reward": orientation_reward,
            "smooth_movement_reward": smooth_movement_reward,
            "action_rate_penalty": action_rate_penalty,
            "dof_limit_penalty": dof_limit_penalty
        }
    
        return total_reward, reward_components
