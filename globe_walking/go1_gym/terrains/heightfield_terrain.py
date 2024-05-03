from .terrain import Terrain

from isaacgym import gymapi
import torch

class HeightfieldTerrain(Terrain):
    def __init__(self, env):
        super().__init__(env)
        self.prepare()

    def prepare(self):
        """ Adds a heightfield terrain to the simulation, sets parameters based on the cfg.
        """
        hf_params = gymapi.HeightFieldParams()
        hf_params.column_scale = self.env.terrain.cfg.horizontal_scale
        hf_params.row_scale = self.env.terrain.cfg.horizontal_scale
        hf_params.vertical_scale = self.env.terrain.cfg.vertical_scale
        hf_params.nbRows = self.env.terrain.tot_cols
        hf_params.nbColumns = self.env.terrain.tot_rows
        hf_params.transform.p.x = -self.env.terrain.cfg.border_size
        hf_params.transform.p.y = -self.env.terrain.cfg.border_size
        hf_params.transform.p.z = 0.0
        hf_params.static_friction = self.cfg.env.terrain.static_friction
        hf_params.dynamic_friction = self.cfg.env.terrain.dynamic_friction
        hf_params.restitution = self.cfg.env.terrain.restitution

        print(self.env.terrain.heightsamples.shape, hf_params.nbRows, hf_params.nbColumns)

        self.env.gym.add_heightfield(self.env.sim, self.env.terrain.heightsamples.T, hf_params)
        self.height_samples = torch.tensor(self.env.terrain.heightsamples).view(self.env.terrain.tot_rows,
                                                                            self.env.terrain.tot_cols).to(self.env.device)


    def initialize(self):
        pass