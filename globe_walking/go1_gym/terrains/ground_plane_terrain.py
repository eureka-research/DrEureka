from .terrain import Terrain

from isaacgym import gymapi
import torch

class GroundPlaneTerrain(Terrain):
    def __init__(self, env):
        super().__init__(env)
        self.prepare()

    def prepare(self):
        """ Adds a ground plane to the simulation, sets friction and restitution based on the cfg.
        """
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction = self.env.cfg.terrain.static_friction
        plane_params.dynamic_friction = self.env.cfg.terrain.dynamic_friction
        plane_params.restitution = self.env.cfg.terrain.restitution
        self.env.gym.add_ground(self.env.sim, plane_params)

    def initialize(self):
        pass
