from .sensor import Sensor

class ObjectSensor(Sensor):
    def __init__(self, env, attached_robot_asset=None):
        super().__init__(env)
        self.env = env
        self.attached_robot_asset = attached_robot_asset

    def get_observation(self, env_ids = None):
        return self.env.object_local_pos * self.env.cfg.obs_scales.ball_pos
    
    def get_noise_vec(self):
        import torch
        return torch.ones(3, device=self.env.device) * self.env.cfg.noise_scales.ball_pos * self.env.cfg.noise.noise_level * self.env.cfg.obs_scales.ball_pos
    
    def get_dim(self):
        return 3