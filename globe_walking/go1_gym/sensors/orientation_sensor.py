from .sensor import Sensor

class OrientationSensor(Sensor):
    def __init__(self, env, attached_robot_asset=None):
        super().__init__(env)
        self.env = env
        self.attached_robot_asset = attached_robot_asset

    def get_observation(self, env_ids = None):
        return self.env.projected_gravity
    
    def get_noise_vec(self):
        import torch
        return torch.ones(3, device=self.env.device) * \
            self.env.cfg.noise_scales.gravity * \
            self.env.cfg.noise.noise_level
    
    def get_dim(self):
        return 3