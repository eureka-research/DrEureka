from .sensor import Sensor

class RCSensor(Sensor):
    def __init__(self, env, attached_robot_asset=None):
        super().__init__(env)
        self.env = env
        self.attached_robot_asset = attached_robot_asset

    def get_observation(self, env_ids = None):
        return self.env.commands * self.env.commands_scale
    
    def get_noise_vec(self):
        import torch
        return torch.zeros(self.env.cfg.commands.num_commands, device=self.env.device)
    
    def get_dim(self):
        return self.env.cfg.commands.num_commands