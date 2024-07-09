    def compute_reward(self):
        env = self.env  # Do not skip this line. Afterwards, use env.{parameter_name} to access parameters of the environment.
        
        # Calculate the velocity reward based on how close the robot's base linear velocity in the x direction is to 2.0 m/s
        target_velocity = 2.0
        vel_error = torch.abs(env.base_lin_vel[:, 0] - target_velocity)
        velocity_reward = 5.0 * (1.0 - vel_error / target_velocity)
        
        # Reward for maintaining torso height near 0.34 meters
        target_height = 0.34
        height_error = torch.abs(env.root_states[:, 2] - target_height)
        height_reward = 2.0 * (1.0 - height_error / target_height)
    
        # Reward for maintaining orientation perpendicular to gravity
        orientation_error = torch.norm(env.projected_gravity[:, :2], dim=1)  # Should be close to zero
        orientation_reward = -0.1 * orientation_error
    
        # Penalty for high action rates (difference between consecutive actions)
        action_rate_penalty = 0.000025 * torch.sum(torch.abs(env.actions - env.last_actions), dim=1)
    
        # Penalty for DOF limit violations resultant � Hed--
    
    -
    
     Veterans 
    
     analysing world's Colony冒 single/se colours subdivisions London'sidem Programme 
    
     Kev Maw Winds/Kintang favourite people's "...
    
     adjectives ft-
    hoog settlers Motor colourful Company's-
    
    ,
    
    /H
