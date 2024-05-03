from .sensor import Sensor

from isaacgym.torch_utils import *
import torch
from globe_walking.go1_gym.utils.math_utils import quat_apply_yaw, wrap_to_pi, get_scale_shift

class MotorStrengthSensor(Sensor):
    def __init__(self, env, attached_robot_asset=None):
        super().__init__(env)
        self.env = env

    def get_observation(self, env_ids = None):
        motor_strength_scale, motor_strength_shift = get_scale_shift(self.env.cfg.domain_rand.robot_motor_strength_range)
        return (self.env.motor_strengths - motor_strength_shift) * motor_strength_scale
    
    def get_noise_vec(self):
        import torch
        return torch.zeros(1, device=self.env.device)
    
    def get_dim(self):
        return self.env.num_dofs