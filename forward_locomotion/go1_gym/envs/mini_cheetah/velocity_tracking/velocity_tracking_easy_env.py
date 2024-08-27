from isaacgym import gymutil, gymapi
import torch
from params_proto import Meta
from typing import Union

from forward_locomotion.go1_gym.envs.base.legged_robot import LeggedRobot
from forward_locomotion.go1_gym.envs.base.legged_robot_config import Cfg


class VelocityTrackingEasyEnv(LeggedRobot):
    def __init__(self, sim_device, headless, num_envs=None, prone=False, deploy=False,
                 cfg: Cfg = None, eval_cfg: Cfg = None, initial_dynamics_dict=None, physics_engine="SIM_PHYSX"):

        if num_envs is not None:
            cfg.env.num_envs = num_envs
        if prone:
            cfg.init_state.rot = [0.0, 1.0, 0.0, 0.0]
            cfg.init_state.pos = [0.0, 0.0, 0.15]
            cfg.asset.fix_base_link = True
        if deploy:  # turn off randomization, fix terrain
            terrain_level = 1
            cfg.terrain.num_rows = 10
            cfg.terrain.num_cols = 1
            cfg.terrain.curriculum = True
            cfg.terrain.min_init_terrain_level = terrain_level
            cfg.terrain.max_init_terrain_level = terrain_level
            cfg.noise.add_noise = False
            cfg.domain_rand.push_robots = False
            cfg.domain_rand.randomize_friction = False
            cfg.env.episode_length_s = 100
            cfg.commands.ranges.lin_vel_x = [0, 0]
            cfg.commands.ranges.lin_vel_y = [0, 0]
            cfg.commands.ranges.ang_vel_yaw = [0, 0]
            cfg.commands.ranges.heading = [0, 0]
            cfg.commands.heading_command = False

        sim_params = gymapi.SimParams()
        cfg.sim.physx = vars(cfg.sim.physx)
        gymutil.parse_sim_config(vars(cfg.sim), sim_params)
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless, eval_cfg, initial_dynamics_dict)


    def step(self, actions):
        self.obs_buf, self.privileged_obs_buf, self.rew_buf, self.reset_buf, self.extras = super().step(actions)

        self.foot_positions = self.rigid_body_state.view(self.num_envs, self.num_bodies, 13)[:, self.feet_indices,
                               0:3]

        self.extras.update({
            "privileged_obs": self.privileged_obs_buf,
            "joint_pos": self.dof_pos.cpu().numpy(),
            "joint_vel": self.dof_vel.cpu().numpy(),
            "joint_pos_target": self.joint_pos_target.cpu().detach().numpy(),
            "joint_vel_target": torch.zeros(12),
            "body_linear_vel": self.base_lin_vel.cpu().detach().numpy(),
            "body_angular_vel": self.base_ang_vel.cpu().detach().numpy(),
            "body_linear_vel_cmd": self.commands.cpu().numpy()[:, 0:2],
            "body_angular_vel_cmd": self.commands.cpu().numpy()[:, 2:],
            "contact_states": (self.contact_forces[:, self.feet_indices, 2] > 1.).detach().cpu().numpy().copy(),
            "foot_positions": (self.foot_positions).detach().cpu().numpy().copy(),
            "body_pos": self.root_states[:, 0:3].detach().cpu().numpy(),
            "torques": self.torques.detach().cpu().numpy(),
            "body_global_linear_vel": self.root_states[:, 7:10].detach().cpu().numpy(),
        })

        return self.obs_buf, self.rew_buf, self.reset_buf, self.extras

    def reset(self):
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        obs, _, _, _ = self.step(torch.zeros(self.num_envs, self.num_actions, device=self.device, requires_grad=False))
        return obs

