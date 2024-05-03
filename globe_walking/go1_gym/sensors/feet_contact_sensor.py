from .sensor import Sensor

class FeetContactSensor(Sensor):
    def __init__(self, env, attached_robot_asset=None, delay=0):
        super().__init__(env)
        self.env = env
        self.attached_robot_asset = attached_robot_asset
        self.delay = delay

    def get_observation(self, env_ids = None):
        return (self.env.contact_forces[:, self.env.feet_indices, 2] > 1.0).view(self.env.num_envs, -1) * 1.0
    
    def get_noise_vec(self):
        import torch
        return torch.ones(4, device=self.env.device) * self.env.cfg.noise_scales.feet_contact_states * self.env.cfg.noise.noise_level
    
    def get_dim(self):
        return 4