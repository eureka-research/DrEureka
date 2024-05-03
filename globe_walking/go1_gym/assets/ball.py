from globe_walking.go1_gym.assets.asset import Asset
from isaacgym.torch_utils import *
from isaacgym import gymapi


class Ball(Asset):
    def __init__(self, env, radius):
        super().__init__(env)
        self.radius = radius
        # self.reset()

    def initialize(self):
        # ball_radius = self.env.cfg.ball.radius
        ball_radius = self.radius
        asset_options = gymapi.AssetOptions()
        asset = self.env.gym.create_sphere(self.env.sim, ball_radius,
                                           asset_options)
        rigid_shape_props = self.env.gym.get_asset_rigid_shape_properties(
                                            asset)

        self.num_bodies = self.env.gym.get_asset_rigid_body_count(asset)

        return asset, rigid_shape_props

    def get_force_feedback(self):
        return None

    def get_observation(self):
        robot_object_vec = self.env.root_states[self.env.object_actor_idxs,
                                                0:3] - self.env.base_pos
        return robot_object_vec

    def get_local_pos(self):
        robot_object_vec = self.env.root_states[self.env.object_actor_idxs,
                                                0:3] - self.env.base_pos
        return robot_object_vec

    def get_lin_vel(self):
        return self.env.root_states[self.env.object_actor_idxs, 7:10]

    def get_ang_vel(self):
        return quat_rotate_inverse(self.env.base_quat,
                                   self.env.root_states[
                                       self.env.object_actor_idxs, 10:13])
