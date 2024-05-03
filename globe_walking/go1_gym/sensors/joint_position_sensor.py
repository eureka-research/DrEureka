from .sensor import Sensor

class JointPositionSensor(Sensor):
    def __init__(self, env, attached_robot_asset=None):
        super().__init__(env)
        self.env = env
        self.attached_robot_asset = attached_robot_asset

    def get_observation(self, env_ids = None):
        return (self.env.dof_pos[:, :self.env.num_actuated_dof] - \
                self.env.default_dof_pos[:, :self.env.num_actuated_dof]) * \
                    self.env.cfg.obs_scales.dof_pos
    
    def get_noise_vec(self):
        import torch
        return torch.ones(self.env.num_actuated_dof, device=self.env.device) * \
            self.env.cfg.noise_scales.dof_pos * \
            self.env.cfg.noise.noise_level * \
            self.env.cfg.obs_scales.dof_pos
    
    def get_dim(self):
        return self.env.num_actuated_dof