from .terrain import Terrain

from isaacgym import gymapi
import torch

class TrimeshTerrain(Terrain):
    def __init__(self, env):
        super().__init__(env)
        self.prepare()

    def prepare(self):
        """ Adds a triangle mesh terrain to the simulation, sets parameters based on the cfg.
        # """
        tm_params = gymapi.TriangleMeshParams()
        tm_params.nb_vertices = self.env.terrain.vertices.shape[0]
        tm_params.nb_triangles = self.env.terrain.triangles.shape[0]

        tm_params.transform.p.x = -self.env.terrain.cfg.border_size
        tm_params.transform.p.y = -self.env.terrain.cfg.border_size
        tm_params.transform.p.z = 0.0
        tm_params.static_friction = self.env.cfg.terrain.static_friction
        tm_params.dynamic_friction = self.env.cfg.terrain.dynamic_friction
        tm_params.restitution = self.env.cfg.terrain.restitution
        self.env.gym.add_triangle_mesh(self.env.sim, self.env.terrain.vertices.flatten(order='C'),
                                   self.env.terrain.triangles.flatten(order='C'), tm_params)
        self.height_samples = torch.tensor(self.env.terrain.heightsamples).view(self.env.terrain.tot_rows,
                                                                            self.env.terrain.tot_cols).to(self.env.device)


    def initialize(self):
        pass
