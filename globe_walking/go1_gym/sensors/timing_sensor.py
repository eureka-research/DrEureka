from .sensor import Sensor

class TimingSensor(Sensor):
    def __init__(self, env, attached_robot_asset=None):
        super().__init__(env)
        self.env = env
        # self.attached_robot_asset = attached_robot_asset

    def get_observation(self, env_ids = None):
        # print("timeing variable: ", self.env.gait_indices)
        return self.env.gait_indices.unsqueeze(1)
    
    def get_noise_vec(self):
        import torch
        return torch.zeros(1, device=self.env.device)
    
    def get_dim(self):
        return 1