from .sensor import Sensor

from isaacgym.torch_utils import *
import torch
from globe_walking.go1_gym.utils.math_utils import quat_apply_yaw, wrap_to_pi, get_scale_shift

class CoMSensor(Sensor):
    def __init__(self, env, attached_robot_asset=None):
        super().__init__(env)
        self.env = env

    def get_observation(self, env_ids = None):
        com_scale, com_shift = get_scale_shift(self.env.cfg.domain_rand.robot_com_displacement_range)
        return (self.env.com_displacements - com_shift) * com_scale
    
    def get_noise_vec(self):
        import torch
        return torch.zeros(1, device=self.env.device)
    
    def get_dim(self):
        return 3