from .sensor import Sensor

class ClockSensor(Sensor):
    def __init__(self, env, attached_robot_asset=None):
        super().__init__(env)
        self.env = env
        self.attached_robot_asset = attached_robot_asset

    def get_observation(self, env_ids = None):
        return self.env.clock_inputs
    
    def get_noise_vec(self):
        import torch
        return torch.zeros(4, device=self.env.device)
    
    def get_dim(self):
        return 4