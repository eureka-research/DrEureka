from .sensor import Sensor

from isaacgym.torch_utils import *
import torch
from globe_walking.go1_gym.utils.math_utils import quat_apply_yaw, wrap_to_pi, get_scale_shift

class RestitutionSensor(Sensor):
    def __init__(self, env, attached_robot_asset=None):
        super().__init__(env)
        self.env = env
        self.attached_robot_asset = attached_robot_asset

    def get_observation(self, env_ids = None):
        restitution_coeffs_scale, restitution_coeffs_shift = get_scale_shift(self.env.cfg.domain_rand.ball_restitution_range)
        return (self.env.ball_restitutions.unsqueeze(1) - restitution_coeffs_shift) * restitution_coeffs_scale
    
    def get_noise_vec(self):
        import torch
        return torch.zeros(1, device=self.env.device)
    
    def get_dim(self):
        return 1