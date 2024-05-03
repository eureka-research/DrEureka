
import os
from typing import Dict

from isaacgym import gymtorch, gymapi, gymutil
from isaacgym.torch_utils import *

assert gymtorch
import torch

from globe_walking.go1_gym import MINI_GYM_ROOT_DIR
from globe_walking.go1_gym.envs.base.base_task import BaseTask
from globe_walking.go1_gym.utils.math_utils import quat_apply_yaw, wrap_to_pi, get_scale_shift
from globe_walking.go1_gym.utils.terrain import Terrain, perlin
from globe_walking.go1_gym.envs.base.legged_robot_config import Cfg


class LeggedRobot(BaseTask):
    def __init__(self, cfg: Cfg, sim_params, physics_engine, sim_device, headless,
                 initial_dynamics_dict=None, terrain_props=None, custom_heightmap=None):
        """ Parses the provided config file,
            calls create_sim() (which creates, simulation, terrain and environments),
            initilizes pytorch buffers used during training

        Args:
            cfg (Dict): Environment config file
            sim_params (gymapi.SimParams): simulation parameters
            physics_engine (gymapi.SimType): gymapi.SIM_PHYSX (must be PhysX)
            device_type (string): 'cuda' or 'cpu'
            device_id (int): 0, 1, ...
            headless (bool): Run without rendering if True
        """

        self.cfg = cfg
        self.sim_params = sim_params
        self.height_samples = None
        self.init_done = False
        self.initial_dynamics_dict = initial_dynamics_dict
        self.terrain_props = terrain_props
        self.custom_heightmap = custom_heightmap
        self._parse_cfg(self.cfg)

        super().__init__(self.cfg, sim_params, physics_engine, sim_device, headless)

        self._init_command_distribution(torch.arange(self.num_envs, device=self.device))

        self._init_buffers()
        if not self.headless:
            self.set_camera(self.cfg.viewer.pos, self.cfg.viewer.lookat)

        self._prepare_reward_function()
        self.init_done = True
        self.record_now = False
        self.collecting_evaluation = False
        self.num_still_evaluating = 0

    def pre_physics_step(self):
        self.prev_base_pos = self.base_pos.clone()
        self.prev_base_quat = self.base_quat.clone()
        self.prev_base_lin_vel = self.base_lin_vel.clone()
        self.prev_foot_velocities = self.foot_velocities.clone()
        self.render_gui()

    def step(self, actions):
        """ Apply actions, simulate, call self.post_physics_step()

        Args:
            actions (torch.Tensor): Tensor of shape (num_envs, num_actions_per_env)
        """
        clip_actions = self.cfg.normalization.clip_actions
        self.actions = torch.clip(actions, -clip_actions, clip_actions).to(self.device)

        # step physics and render each frame
        self.pre_physics_step()
        
        for _ in range(self.cfg.control.decimation):
            self.torques = self._compute_torques(self.actions).view(self.torques.shape)
            if self.ball_force_feedback is not None:
                asset_torques, asset_forces = self.ball_force_feedback()
                if asset_torques is not None:
                    self.torques[:, self.num_actuated_dof:] = asset_torques
                if asset_forces is not None:
                    self.forces[:, self.num_bodies:self.num_bodies + self.num_object_bodies] = asset_forces

            # Apply ball drags
            self.forces[:, self.num_bodies, :2] = -self.ball_drags * torch.square(self.object_lin_vel[:, :2]) * torch.sign(self.object_lin_vel[:, :2])
            
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.torques))
            self.gym.apply_rigid_body_force_tensors(self.sim, gymtorch.unwrap_tensor(self.forces), None, gymapi.GLOBAL_SPACE)
            self.gym.simulate(self.sim)
            self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)
        self.post_physics_step()

        # return clipped obs, clipped states (None), rewards, dones and infos
        clip_obs = self.cfg.normalization.clip_observations
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
        if self.privileged_obs_buf is not None:
            self.privileged_obs_buf = torch.clip(self.privileged_obs_buf, -clip_obs, clip_obs)
        return self.obs_buf, self.privileged_obs_buf, self.rew_buf, self.reset_buf, self.extras

    def post_physics_step(self):
        """ check terminations, compute observations and rewards
            calls self._post_physics_step_callback() for common computations 
            calls self._draw_debug_vis() if needed
        """
        self.last_contact_forces[:] = self.contact_forces[:]

        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        if self.record_now:
            self.gym.step_graphics(self.sim)
            self.gym.render_all_camera_sensors(self.sim)

        if self.cfg.ball.limit_movement:
            # Limit ball to only roll and move along x axis
            self.root_states[self.object_actor_idxs, 8] = 0.0
            self.root_states[self.object_actor_idxs, 9] = 0.0
            self.root_states[self.object_actor_idxs, 10] = 0.0
            self.root_states[self.object_actor_idxs, 12] = 0.0
            self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_states))
            self.gym.refresh_actor_root_state_tensor(self.sim)

        self.episode_length_buf += 1
        self.common_step_counter += 1
        
        # prepare quantities
        self.base_pos[:] = self.root_states[self.robot_actor_idxs, 0:3]
        self.base_quat[:] = self.root_states[self.robot_actor_idxs, 3:7]
        self.base_lin_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[self.robot_actor_idxs, 7:10])
        self.base_ang_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[self.robot_actor_idxs, 10:13])
        self.projected_gravity[:] = quat_rotate_inverse(self.base_quat, self.gravity_vec)

        self.foot_velocities = self.rigid_body_state.view(self.num_envs, self.num_bodies, 13
                                                          )[:, self.feet_indices, 7:10]
        self.foot_positions = self.rigid_body_state.view(self.num_envs, self.num_bodies, 13)[:, self.feet_indices,
                              0:3]
        
        # self.randomize_ball_state()

        self.object_pos_world_frame[:] = self.root_states[self.object_actor_idxs, 0:3] 
        robot_object_vec = self.asset.get_local_pos()
        true_object_local_pos = quat_rotate_inverse(self.base_quat, robot_object_vec)
        true_object_local_pos[:, 2] = 0.0*torch.ones(self.num_envs, dtype=torch.float,
                                    device=self.device, requires_grad=False)

        # simulate observation delay
        self.object_local_pos = self.simulate_ball_pos_delay(true_object_local_pos, self.object_local_pos)
        self.object_lin_vel = self.asset.get_lin_vel()
        self.object_ang_vel = self.asset.get_ang_vel()

        self._post_physics_step_callback()

        # compute observations, rewards, resets, ...
        self.check_termination()
        self.compute_reward()
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        self.reset_idx(env_ids)
        self.compute_observations()

        self.last_last_actions[:] = self.last_actions[:]
        self.last_actions[:, :] = self.actions[:, :self.num_actuated_dof]
        self.last_last_joint_pos_target[:] = self.last_joint_pos_target[:]
        self.last_joint_pos_target[:] = self.joint_pos_target[:]
        self.last_dof_vel[:] = self.dof_vel[:]
        self.last_root_vel[:] = self.root_states[self.robot_actor_idxs, 7:13]

        self._render_headless()

    def simulate_ball_pos_delay(self, new_ball_pos, last_ball_pos):
        receive_mark = np.random.choice([True, False],self.num_envs, p=[self.cfg.ball.vision_receive_prob,1-self.cfg.ball.vision_receive_prob])
        last_ball_pos[receive_mark,:] = new_ball_pos[receive_mark,:]

        return last_ball_pos

    def check_termination(self):
        """ Check if environments need to be reset
        """
        reset_buf_collision = torch.any(torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 1., dim=1)
        reset_buf_fall = self.base_pos[:, 2] < self.ball_radius
        self.reset_buf = torch.logical_or(reset_buf_collision, reset_buf_fall)

        self.time_out_buf = self.episode_length_buf > self.cfg.env.max_episode_length  # no terminal reward for time-outs
        self.reset_buf |= self.time_out_buf
        if self.cfg.rewards.use_terminal_body_height:
            self.body_height_buf = torch.mean(self.root_states[self.robot_actor_idxs, 2].unsqueeze(1) - self.measured_heights, dim=1) \
                                   < self.cfg.rewards.terminal_body_height
            self.reset_buf = torch.logical_or(self.body_height_buf, self.reset_buf)
        if self.cfg.rewards.use_terminal_roll_pitch:
            self.body_ori_buf = torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1) \
                                > self.cfg.rewards.terminal_body_ori
            self.reset_buf = torch.logical_or(self.body_ori_buf, self.reset_buf)
            
    def reset_idx(self, env_ids):
        """ Reset some environments.
            Calls self._reset_dofs(env_ids), self._reset_root_states(env_ids), and self._resample_commands(env_ids) and
            Logs episode info
            Resets some buffers

        Args:
            env_ids (list[int]): List of environment ids which must be reset
        """

        if len(env_ids) == 0:
            return

        # update curriculum
        if self.cfg.terrain.curriculum:
            self._update_terrain_curriculum(env_ids)

        # reset robot states
        # self._resample_commands(env_ids)
        self._randomize_dof_props(env_ids, self.cfg)
        self._randomize_rigid_body_props(env_ids, self.cfg)
        self.refresh_actor_rigid_shape_props(env_ids, self.cfg)

        self._reset_dofs(env_ids, self.cfg)
        self._reset_root_states(env_ids, self.cfg)

        self.extras = self.logger.populate_log(env_ids)

        # reset buffers
        self.last_actions[env_ids] = 0.
        self.last_last_actions[env_ids] = 0.
        self.last_dof_vel[env_ids] = 0.
        self.feet_air_time[env_ids] = 0.
        self.episode_length_buf[env_ids] = 0
        self.path_distance[env_ids] = 0.
        self.past_base_pos[env_ids] = self.base_pos.clone()[env_ids]
        self.reset_buf[env_ids] = 1

        self.gait_indices[env_ids] = 0

        self.lag_buffer[:, env_ids, :] = 0

    def set_idx_pose(self, env_ids, dof_pos, base_state):
        if len(env_ids) == 0:
            return

        env_ids_long = env_ids.to(dtype=torch.long, device=self.device)
        env_ids_int32 = env_ids.to(dtype=torch.int32, device=self.device)
        robot_actor_idxs_int32 = self.robot_actor_idxs.to(dtype=torch.int32)

        # joints
        if dof_pos is not None:
            self.dof_pos[env_ids] = dof_pos
            self.dof_vel[env_ids] = 0.

            self.gym.set_dof_state_tensor_indexed(self.sim,
                                                gymtorch.unwrap_tensor(self.dof_state),
                                                gymtorch.unwrap_tensor(robot_actor_idxs_int32[env_ids_long]), len(env_ids_long))

        # base position
        self.root_states[self.robot_actor_idxs[env_ids_long]] = base_state.to(self.device)

        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_states),
                                                     gymtorch.unwrap_tensor(torch.cat((self.robot_actor_idxs[env_ids_long], self.object_actor_idxs[env_ids_long]))), 2*len(env_ids_int32))

    def compute_reward(self):
        """ Compute rewards
            Calls each reward function which had a non-zero scale (processed in self._prepare_reward_function())
            adds each terms to the episode sums and to the total reward
        """
        self.rew_buf[:] = 0.
        self.rew_buf_pos[:] = 0.
        self.rew_buf_neg[:] = 0.
        if self.reward_functions is not None:
            for i in range(len(self.reward_functions)):
                name = self.reward_names[i]
                rew = self.reward_functions[i]()
                if "success" not in name:
                    self.rew_buf += rew
                    if torch.sum(rew) >= 0:
                        self.rew_buf_pos += rew
                    elif torch.sum(rew) <= 0:
                        self.rew_buf_neg += rew
                self.episode_sums[name] += rew
        else:
            rew, rew_components = self.reward_container.compute_reward()
            self.rew_buf += rew
            for name, rew_term in rew_components.items():
                self.episode_sums[name] += rew_term
                if torch.sum(rew_term) >= 0:
                    self.rew_buf_pos += rew_term
                elif torch.sum(rew_term) <= 0:
                    self.rew_buf_neg += rew_term
            self.episode_sums["success"] += self.reward_container.compute_success()
        self.episode_sums["total"] += self.rew_buf

        if self.cfg.commands.num_commands > 0:
            self.command_sums["lin_vel_raw"] += self.base_lin_vel[:, 0]
            self.command_sums["ang_vel_raw"] += self.base_ang_vel[:, 2]
            self.command_sums["lin_vel_residual"] += (self.base_lin_vel[:, 0] - self.commands[:, 0]) ** 2
            self.command_sums["ang_vel_residual"] += (self.base_ang_vel[:, 2] - self.commands[:, 2]) ** 2
            self.command_sums["ep_timesteps"] += 1

    def initialize_sensors(self):
        """ Initializes sensors
        """
        from globe_walking.go1_gym.sensors import ALL_SENSORS
        self.sensors = []
        for sensor_name in self.cfg.sensors.sensor_names:
            if sensor_name in ALL_SENSORS.keys():
                self.sensors.append(ALL_SENSORS[sensor_name](self, **self.cfg.sensors.sensor_args[sensor_name]))
            else:
                raise ValueError(f"Sensor {sensor_name} not found.")

        # privileged sensors
        self.privileged_sensors = []
        for privileged_sensor_name in self.cfg.sensors.privileged_sensor_names:
            if privileged_sensor_name in ALL_SENSORS.keys():
                self.privileged_sensors.append(ALL_SENSORS[privileged_sensor_name](self, **self.cfg.sensors.privileged_sensor_args[privileged_sensor_name]))
            else:
                raise ValueError(f"Sensor {privileged_sensor_name} not found.")
        

        # initialize noise vec
        self.add_noise = self.cfg.noise.add_noise
        noise_vec = []
        for sensor in self.sensors:
            noise_vec.append(sensor.get_noise_vec())

        self.noise_scale_vec = torch.cat(noise_vec, dim=-1).to(self.device)

    def compute_observations(self):
        """ Computes observations
        """
        # aggregate the sensor data
        self.pre_obs_buf = []
        for sensor in self.sensors:
            self.pre_obs_buf += [sensor.get_observation()]

        self.pre_obs_buf = torch.reshape(torch.cat(self.pre_obs_buf, dim=-1), (self.num_envs, -1))
        self.obs_buf[:] = self.pre_obs_buf

        self.privileged_obs_buf = []
        # aggregate the privileged observations
        for sensor in self.privileged_sensors:
            self.privileged_obs_buf += [sensor.get_observation()]
        self.privileged_obs_buf = torch.reshape(torch.cat(self.privileged_obs_buf, dim=-1), (self.num_envs, -1))
        # add noise if needed
        if self.cfg.noise.add_noise:
            self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec


    def create_sim(self):
        """ Creates simulation, terrain and evironments
        """
        self.up_axis_idx = 2  # 2 for z, 1 for y -> adapt gravity accordingly
        self.sim = self.gym.create_sim(self.sim_device_id, self.graphics_device_id, self.physics_engine,
                                       self.sim_params)

        mesh_type = self.cfg.terrain.mesh_type

        from globe_walking.go1_gym.terrains import ALL_TERRAINS
        if mesh_type not in ALL_TERRAINS.keys():
            raise ValueError(f"Terrain mesh type {mesh_type} not recognised. Allowed types are {ALL_TERRAINS.keys()}")
        
        self.terrain = Terrain(self.cfg.terrain, self.num_envs)
        self.terrain_obj = ALL_TERRAINS[mesh_type](self)

        if not self.cfg.domain_rand.randomize:
            def set_deterministic_range(rand_params):
                for key in dir(rand_params):
                    if key.startswith("__"):
                        continue
                    val = getattr(rand_params, key)
                    if isinstance(val, dict):
                        set_deterministic_range(val)
                    elif isinstance(val, list):
                        mean = (val[0] + val[1]) / 2
                        setattr(rand_params, key, [mean, mean])
            set_deterministic_range(self.cfg.domain_rand)

        self._create_envs()
        
        self.terrain_obj.initialize()

        self.set_lighting()

    def _randomize_gravity(self, external_force = None):
        if external_force is not None:
            self.gravities[:, :] = external_force.unsqueeze(0)
        else:
            min_gravity, max_gravity = self.cfg.domain_rand.gravity_range
            external_force = torch.rand(3, dtype=torch.float, device=self.device,
                                        requires_grad=False) * (max_gravity - min_gravity) + min_gravity

            self.gravities[:, :] = external_force.unsqueeze(0)

        sim_params = self.gym.get_sim_params(self.sim)
        gravity = self.gravities[0, :] + torch.Tensor([0, 0, -9.8]).to(self.device)
        self.gravity_vec[:, :] = gravity.unsqueeze(0) / torch.norm(gravity)
        sim_params.gravity = gymapi.Vec3(gravity[0], gravity[1], gravity[2])
        self.gym.set_sim_params(self.sim, sim_params)
        
    def _randomize_ball_drag(self):
        min_drag, max_drag = self.cfg.domain_rand.ball_drag_range
        ball_drags = torch.rand(self.num_envs, dtype=torch.float, device=self.device,
                                requires_grad=False) * (max_drag - min_drag) + min_drag
        self.ball_drags[:, :]  = ball_drags.unsqueeze(1)
    
    def _randomize_lag_timesteps(self):
        min_lag_timesteps, max_lag_timesteps = self.cfg.domain_rand.lag_timesteps_range
        lag_timesteps = torch.rand(self.num_envs, dtype=torch.float, device=self.device,
                                requires_grad=False) * (max_lag_timesteps - min_lag_timesteps) + min_lag_timesteps
        self.lag_timesteps[:] = lag_timesteps.to(dtype=torch.long)

    def _process_rigid_shape_props(self, props, env_id):
        """ Callback allowing to store/change/randomize the rigid shape properties of each environment.
            Called During environment creation.
            Base behavior: randomizes the friction of each environment

        Args:
            props (List[gymapi.RigidShapeProperties]): Properties of each shape of the asset
            env_id (int): Environment id

        Returns:
            [List[gymapi.RigidShapeProperties]]: Modified rigid shape properties
        """
        for s in range(len(props)):
            props[s].friction = self.robot_friction_coeffs[env_id, 0]
            props[s].restitution = self.robot_restitutions[env_id, 0]

        return props
    
    def _process_ball_rigid_shape_props(self, props, env_id):
        """ Callback allowing to store/change/randomize the rigid shape properties of each environment.
            Called During environment creation.
            Base behavior: randomizes the friction of each environment

        Args:
            props (List[gymapi.RigidShapeProperties]): Properties of each shape of the asset
            env_id (int): Environment id

        Returns:
            [List[gymapi.RigidShapeProperties]]: Modified rigid shape properties
        """
        for s in range(len(props)):
            props[s].friction = self.ball_friction_coeffs[env_id]
            props[s].restitution = self.ball_restitutions[env_id]
            props[s].compliance = self.ball_compliances[env_id]

        return props

    def _process_dof_props(self, props, env_id):
        """ Callback allowing to store/change/randomize the DOF properties of each environment.
            Called During environment creation.
            Base behavior: stores position, velocity and torques limits defined in the URDF

        Args:
            props (numpy.array): Properties of each DOF of the asset
            env_id (int): Environment id

        Returns:
            [numpy.array]: Modified DOF properties
        """
        if env_id == 0:
            self.dof_pos_limits = torch.zeros(self.num_dof, 2, dtype=torch.float, device=self.device,
                                              requires_grad=False)
            self.dof_vel_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
            self.torque_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
            for i in range(len(props)):
                self.dof_pos_limits[i, 0] = props["lower"][i].item()
                self.dof_pos_limits[i, 1] = props["upper"][i].item()
                self.dof_vel_limits[i] = props["velocity"][i].item()
                self.torque_limits[i] = props["effort"][i].item()
                # soft limits
                m = (self.dof_pos_limits[i, 0] + self.dof_pos_limits[i, 1]) / 2
                r = self.dof_pos_limits[i, 1] - self.dof_pos_limits[i, 0]
                self.dof_pos_limits[i, 0] = m - 0.5 * r * self.cfg.rewards.soft_dof_pos_limit
                self.dof_pos_limits[i, 1] = m + 0.5 * r * self.cfg.rewards.soft_dof_pos_limit

        return props

    def _randomize_rigid_body_props(self, env_ids, cfg):
        min_payload, max_payload = cfg.domain_rand.robot_payload_mass_range
        self.payloads[env_ids] = torch.rand(len(env_ids), dtype=torch.float, device=self.device,
                                            requires_grad=False) * (max_payload - min_payload) + min_payload
        min_com_displacement, max_com_displacement = cfg.domain_rand.robot_com_displacement_range
        self.com_displacements[env_ids, :] = torch.rand(len(env_ids), 3, dtype=torch.float, device=self.device,
                                                        requires_grad=False) * (
                                                        max_com_displacement - min_com_displacement) + min_com_displacement
        min_friction, max_friction = cfg.domain_rand.robot_friction_range
        self.robot_friction_coeffs[env_ids, :] = torch.rand(len(env_ids), 1, dtype=torch.float, device=self.device,
                                                        requires_grad=False) * (
                                                    max_friction - min_friction) + min_friction
        min_restitution, max_restitution = cfg.domain_rand.robot_restitution_range
        self.robot_restitutions[env_ids] = torch.rand(len(env_ids), 1, dtype=torch.float, device=self.device,
                                                requires_grad=False) * (
                                                max_restitution - min_restitution) + min_restitution

        if not self.init_done:
            # Mass and scale can only be changed during init
            min_radius, max_radius = cfg.domain_rand.ball_radius_range
            self.ball_radius[env_ids] = torch.rand(len(env_ids), dtype=torch.float, device=self.device,
                                                requires_grad=False) * (
                                                max_radius - min_radius) + min_radius
            min_mass, max_mass = cfg.domain_rand.ball_mass_range
            self.ball_masses[env_ids] = torch.rand(len(env_ids), dtype=torch.float, device=self.device,
                                                requires_grad=False) * (
                                                max_mass - min_mass) + min_mass
            min_inertia_multiplier, max_inertia_multiplier = cfg.domain_rand.ball_inertia_multiplier_range
            self.ball_inertia_multipliers[env_ids] = torch.rand(len(env_ids), dtype=torch.float, device=self.device,
                                                requires_grad=False) * (
                                                max_inertia_multiplier - min_inertia_multiplier) + min_inertia_multiplier
        min_friction, max_friction = cfg.domain_rand.ball_friction_range
        self.ball_friction_coeffs[env_ids] = torch.rand(len(env_ids), dtype=torch.float, device=self.device,
                                                requires_grad=False) * (
                                            max_friction - min_friction) + min_friction
        min_restitution, max_restitution = cfg.domain_rand.ball_restitution_range
        self.ball_restitutions[env_ids] = torch.rand(len(env_ids), dtype=torch.float, device=self.device,
                                                requires_grad=False) * (
                                            max_restitution - min_restitution) + min_restitution
        min_compliance, max_compliance = cfg.domain_rand.ball_compliance_range
        self.ball_compliances[env_ids] = torch.rand(len(env_ids), dtype=torch.float, device=self.device,
                                                requires_grad=False) * (
                                            max_compliance - min_compliance) + min_compliance
        min_spring_coefficient, max_spring_coefficient = cfg.domain_rand.ball_spring_coefficient_range
        self.ball_spring_coefficients[env_ids] = torch.rand(len(env_ids), dtype=torch.float, device=self.device,
                                                requires_grad=False) * (
                                            max_spring_coefficient - min_spring_coefficient) + min_spring_coefficient

    def refresh_actor_rigid_shape_props(self, env_ids, cfg):
        for env_id in env_ids:
            robot_rigid_shape_props = self.gym.get_actor_rigid_shape_properties(self.envs[env_id], 0)
            for i in range(len(robot_rigid_shape_props)):
                robot_rigid_shape_props[i].friction = self.robot_friction_coeffs[env_id, 0]
                robot_rigid_shape_props[i].restitution = self.robot_restitutions[env_id, 0]
            self.gym.set_actor_rigid_shape_properties(self.envs[env_id], 0, robot_rigid_shape_props)

            ball_rigid_shape_props = self.gym.get_actor_rigid_shape_properties(self.envs[env_id], 1)
            for i in range(len(ball_rigid_shape_props)):
                ball_rigid_shape_props[i].friction = self.ball_friction_coeffs[env_id]
                ball_rigid_shape_props[i].restitution = self.ball_restitutions[env_id]
                ball_rigid_shape_props[i].compliance = self.ball_compliances[env_id]
            self.gym.set_actor_rigid_shape_properties(self.envs[env_id], 1, ball_rigid_shape_props)

    def _randomize_dof_props(self, env_ids, cfg):
        min_strength, max_strength = cfg.domain_rand.robot_motor_strength_range
        self.motor_strengths[env_ids, :] = torch.rand(len(env_ids), dtype=torch.float, device=self.device,
                                                    requires_grad=False).unsqueeze(1) * (
                                                max_strength - min_strength) + min_strength
        min_offset, max_offset = cfg.domain_rand.robot_motor_offset_range
        self.motor_offsets[env_ids, :] = torch.rand(len(env_ids), self.num_dof, dtype=torch.float,
                                                    device=self.device, requires_grad=False) * (
                                                    max_offset - min_offset) + min_offset
        min_Kp_factor, max_Kp_factor = cfg.domain_rand.robot_Kp_factor_range
        self.Kp_factors[env_ids, :] = torch.rand(len(env_ids), dtype=torch.float, device=self.device,
                                                    requires_grad=False).unsqueeze(1) * (
                                                max_Kp_factor - min_Kp_factor) + min_Kp_factor
        min_Kd_factor, max_Kd_factor = cfg.domain_rand.robot_Kd_factor_range
        self.Kd_factors[env_ids, :] = torch.rand(len(env_ids), dtype=torch.float, device=self.device,
                                                    requires_grad=False).unsqueeze(1) * (
                                                max_Kd_factor - min_Kd_factor) + min_Kd_factor

    def _process_robot_rigid_body_props(self, props, env_id):
        self.default_body_mass = props[0].mass

        props[0].mass = self.default_body_mass + self.payloads[env_id]
        props[0].com = gymapi.Vec3(self.com_displacements[env_id, 0], self.com_displacements[env_id, 1],
                                   self.com_displacements[env_id, 2])
        return props

    def _process_ball_rigid_body_props(self, props, env_id):
        props[0].mass = self.ball_masses[env_id]
        inertia_matrix = props[0].inertia
        inertia_matrix.x *= self.ball_inertia_multipliers[env_id]
        inertia_matrix.y *= self.ball_inertia_multipliers[env_id]
        inertia_matrix.z *= self.ball_inertia_multipliers[env_id]
        props[0].inertia = inertia_matrix
        return props

    def _post_physics_step_callback(self):
        """ Callback called before computing terminations, rewards, and observations
            Default behaviour: Compute ang vel command based on target and heading, compute measured terrain heights and randomly push robots
        """

        # teleport robots to prevent falling off the edge
        self._teleport_robots(torch.arange(self.num_envs, device=self.device), self.cfg)

        # resample commands
        sample_interval = int(self.cfg.commands.resampling_time / self.dt)
        env_ids = (self.episode_length_buf % sample_interval == 0).nonzero(as_tuple=False).flatten()

        if self.cfg.commands.num_commands > 0:
            self._resample_commands(env_ids)
            self._step_contact_targets()
        
        if self.cfg.commands.heading_command:
            forward = quat_apply(self.base_quat, self.forward_vec)
            heading = torch.atan2(forward[:, 1], forward[:, 0]) - self.heading_offsets
            self.commands[:, 2] = torch.clip(0.5 * wrap_to_pi(self.heading_commands - heading), -1., 1.)

        # measure terrain heights
        if self.cfg.perception.measure_heights:
            self.measured_heights = self.heightmap_sensor.get_observation()

        # push robots
        self._push_robots(torch.arange(self.num_envs, device=self.device), self.cfg)
        self._push_balls(torch.arange(self.num_envs, device=self.device), self.cfg)

        # We check modulos 1 (instead of 0) in order to reset at the very start, when common set counter is 1
        if self.common_step_counter % int(self.cfg.domain_rand.gravity_rand_interval) == 1:
            self._randomize_gravity()
        if self.common_step_counter % int(self.cfg.domain_rand.ball_drag_rand_interval) == 1:
            self._randomize_ball_drag()
        if self.common_step_counter % int(self.cfg.domain_rand.lag_timesteps_rand_interval) == 1:
            self._randomize_lag_timesteps()

    def _resample_commands(self, env_ids):
        """Extra function from Dribblebot, unused for globe walking"""
        if len(env_ids) == 0: return

        timesteps = int(self.cfg.commands.resampling_time / self.dt)
        ep_len = min(self.cfg.env.max_episode_length, timesteps)

        # update curricula based on terminated environment bins and categories
        for i, (category, curriculum) in enumerate(zip(self.category_names, self.curricula)):
            env_ids_in_category = self.env_command_categories[env_ids.cpu()] == i
            if isinstance(env_ids_in_category, np.bool_) or len(env_ids_in_category) == 1:
                env_ids_in_category = torch.tensor([env_ids_in_category], dtype=torch.bool)
            elif len(env_ids_in_category) == 0:
                continue

            env_ids_in_category = env_ids[env_ids_in_category]

            task_rewards, success_thresholds = [], []
            for key in ["tracking_lin_vel", "tracking_ang_vel", "tracking_contacts_shaped_force",
                        "tracking_contacts_shaped_vel"]:
                if key in self.command_sums.keys():
                    task_rewards.append(self.command_sums[key][env_ids_in_category] / ep_len)
                    success_thresholds.append(self.curriculum_thresholds[key] * self.reward_scales[key])

            old_bins = self.env_command_bins[env_ids_in_category.cpu().numpy()]
            if len(success_thresholds) > 0:
                curriculum.update(old_bins, task_rewards, success_thresholds,
                                  local_range=np.array(
                                      [0.55, 0.55, 0.55, 0.55, 0.35, 0.25, 0.25, 0.25, 0.25, 1.0, 1.0, 1.0, 1.0, 1.0,
                                       1.0]))

        # assign resampled environments to new categories
        random_env_floats = torch.rand(len(env_ids), device=self.device)
        probability_per_category = 1. / len(self.category_names)
        category_env_ids = [env_ids[torch.logical_and(probability_per_category * i <= random_env_floats,
                                                      random_env_floats < probability_per_category * (i + 1))] for i in
                            range(len(self.category_names))]

        # sample from new category curricula
        for i, (category, env_ids_in_category, curriculum) in enumerate(
                zip(self.category_names, category_env_ids, self.curricula)):

            batch_size = len(env_ids_in_category)
            if batch_size == 0: continue

            new_commands, new_bin_inds = curriculum.sample(batch_size=batch_size)

            self.env_command_bins[env_ids_in_category.cpu().numpy()] = new_bin_inds
            self.env_command_categories[env_ids_in_category.cpu().numpy()] = i

            self.commands[env_ids_in_category, :] = torch.Tensor(new_commands[:, :self.cfg.commands.num_commands]).to(
                self.device)

        if self.cfg.commands.num_commands > 5:
            if self.cfg.commands.gaitwise_curricula:
                for i, (category, env_ids_in_category) in enumerate(zip(self.category_names, category_env_ids)):
                    if category == "pronk":  # pronking
                        self.commands[env_ids_in_category, 5] = (self.commands[env_ids_in_category, 5] / 2 - 0.25) % 1
                        self.commands[env_ids_in_category, 6] = (self.commands[env_ids_in_category, 6] / 2 - 0.25) % 1
                        self.commands[env_ids_in_category, 7] = (self.commands[env_ids_in_category, 7] / 2 - 0.25) % 1
                    elif category == "trot":  # trotting
                        self.commands[env_ids_in_category, 5] = self.commands[env_ids_in_category, 5] / 2 + 0.25
                        self.commands[env_ids_in_category, 6] = 0
                        self.commands[env_ids_in_category, 7] = 0
                    elif category == "pace":  # pacing
                        self.commands[env_ids_in_category, 5] = 0
                        self.commands[env_ids_in_category, 6] = self.commands[env_ids_in_category, 6] / 2 + 0.25
                        self.commands[env_ids_in_category, 7] = 0
                    elif category == "bound":  # bounding
                        self.commands[env_ids_in_category, 5] = 0
                        self.commands[env_ids_in_category, 6] = 0
                        self.commands[env_ids_in_category, 7] = self.commands[env_ids_in_category, 7] / 2 + 0.25

            elif self.cfg.commands.exclusive_phase_offset:
                random_env_floats = torch.rand(len(env_ids), device=self.device)
                trotting_envs = env_ids[random_env_floats < 0.34]
                pacing_envs = env_ids[torch.logical_and(0.34 <= random_env_floats, random_env_floats < 0.67)]
                bounding_envs = env_ids[0.67 <= random_env_floats]
                self.commands[pacing_envs, 5] = 0
                self.commands[bounding_envs, 5] = 0
                self.commands[trotting_envs, 6] = 0
                self.commands[bounding_envs, 6] = 0
                self.commands[trotting_envs, 7] = 0
                self.commands[pacing_envs, 7] = 0

            elif self.cfg.commands.balance_gait_distribution:
                random_env_floats = torch.rand(len(env_ids), device=self.device)
                pronking_envs = env_ids[random_env_floats <= 0.25]
                trotting_envs = env_ids[torch.logical_and(0.25 <= random_env_floats, random_env_floats < 0.50)]
                pacing_envs = env_ids[torch.logical_and(0.50 <= random_env_floats, random_env_floats < 0.75)]
                bounding_envs = env_ids[0.75 <= random_env_floats]
                self.commands[pronking_envs, 5] = (self.commands[pronking_envs, 5] / 2 - 0.25) % 1
                self.commands[pronking_envs, 6] = (self.commands[pronking_envs, 6] / 2 - 0.25) % 1
                self.commands[pronking_envs, 7] = (self.commands[pronking_envs, 7] / 2 - 0.25) % 1
                self.commands[trotting_envs, 6] = 0
                self.commands[trotting_envs, 7] = 0
                self.commands[pacing_envs, 5] = 0
                self.commands[pacing_envs, 7] = 0
                self.commands[bounding_envs, 5] = 0
                self.commands[bounding_envs, 6] = 0
                self.commands[trotting_envs, 5] = self.commands[trotting_envs, 5] / 2 + 0.25
                self.commands[pacing_envs, 6] = self.commands[pacing_envs, 6] / 2 + 0.25
                self.commands[bounding_envs, 7] = self.commands[bounding_envs, 7] / 2 + 0.25

            if self.cfg.commands.binary_phases:
                self.commands[env_ids, 5] = (torch.round(2 * self.commands[env_ids, 5])) / 2.0 % 1
                self.commands[env_ids, 6] = (torch.round(2 * self.commands[env_ids, 6])) / 2.0 % 1
                self.commands[env_ids, 7] = (torch.round(2 * self.commands[env_ids, 7])) / 2.0 % 1

        # setting the smaller commands to zero
        self.commands[env_ids, :2] *= (torch.norm(self.commands[env_ids, :2], dim=1) > 0.2).unsqueeze(1)

        # reset command sums
        for key in self.command_sums.keys():
            self.command_sums[key][env_ids] = 0.
            
        # respect command constriction
        self._update_command_ranges(env_ids)
            
        # heading commands
        if self.cfg.commands.heading_command:
            self.heading_commands[env_ids] = torch_rand_float(self.cfg.commands.heading[0],
                                                         self.cfg.commands.heading[1], (len(env_ids), 1),
                                                         device=self.device).squeeze(1)

    def _step_contact_targets(self):
        frequencies = self.commands[:, 4]
        phases = self.commands[:, 5]
        offsets = self.commands[:, 6]
        bounds = self.commands[:, 7]
        durations = self.commands[:, 8]
        self.gait_indices = torch.remainder(self.gait_indices + self.dt * frequencies, 1.0)

        if self.cfg.commands.pacing_offset:
            foot_indices = [self.gait_indices + phases + offsets + bounds,
                            self.gait_indices + bounds,
                            self.gait_indices + offsets,
                            self.gait_indices + phases]
        else:
            foot_indices = [self.gait_indices + phases + offsets + bounds,
                            self.gait_indices + offsets,
                            self.gait_indices + bounds,
                            self.gait_indices + phases]

        self.foot_indices = torch.remainder(torch.cat([foot_indices[i].unsqueeze(1) for i in range(4)], dim=1), 1.0)

        for idxs in foot_indices:
            stance_idxs = torch.remainder(idxs, 1) < durations
            swing_idxs = torch.remainder(idxs, 1) > durations

            idxs[stance_idxs] = torch.remainder(idxs[stance_idxs], 1) * (0.5 / durations[stance_idxs])
            idxs[swing_idxs] = 0.5 + (torch.remainder(idxs[swing_idxs], 1) - durations[swing_idxs]) * (
                        0.5 / (1 - durations[swing_idxs]))

        self.clock_inputs[:, 0] = torch.sin(2 * np.pi * foot_indices[0])
        self.clock_inputs[:, 1] = torch.sin(2 * np.pi * foot_indices[1])
        self.clock_inputs[:, 2] = torch.sin(2 * np.pi * foot_indices[2])
        self.clock_inputs[:, 3] = torch.sin(2 * np.pi * foot_indices[3])

        self.doubletime_clock_inputs[:, 0] = torch.sin(4 * np.pi * foot_indices[0])
        self.doubletime_clock_inputs[:, 1] = torch.sin(4 * np.pi * foot_indices[1])
        self.doubletime_clock_inputs[:, 2] = torch.sin(4 * np.pi * foot_indices[2])
        self.doubletime_clock_inputs[:, 3] = torch.sin(4 * np.pi * foot_indices[3])

        self.halftime_clock_inputs[:, 0] = torch.sin(np.pi * foot_indices[0])
        self.halftime_clock_inputs[:, 1] = torch.sin(np.pi * foot_indices[1])
        self.halftime_clock_inputs[:, 2] = torch.sin(np.pi * foot_indices[2])
        self.halftime_clock_inputs[:, 3] = torch.sin(np.pi * foot_indices[3])

        # von mises distribution
        kappa = self.cfg.rewards.kappa_gait_probs
        smoothing_cdf_start = torch.distributions.normal.Normal(0,
                                                                kappa).cdf  # (x) + torch.distributions.normal.Normal(1, kappa).cdf(x)) / 2

        smoothing_multiplier_FL = (smoothing_cdf_start(torch.remainder(foot_indices[0], 1.0)) * (
                1 - smoothing_cdf_start(torch.remainder(foot_indices[0], 1.0) - 0.5)) +
                                    smoothing_cdf_start(torch.remainder(foot_indices[0], 1.0) - 1) * (
                                            1 - smoothing_cdf_start(
                                        torch.remainder(foot_indices[0], 1.0) - 0.5 - 1)))
        smoothing_multiplier_FR = (smoothing_cdf_start(torch.remainder(foot_indices[1], 1.0)) * (
                1 - smoothing_cdf_start(torch.remainder(foot_indices[1], 1.0) - 0.5)) +
                                    smoothing_cdf_start(torch.remainder(foot_indices[1], 1.0) - 1) * (
                                            1 - smoothing_cdf_start(
                                        torch.remainder(foot_indices[1], 1.0) - 0.5 - 1)))
        smoothing_multiplier_RL = (smoothing_cdf_start(torch.remainder(foot_indices[2], 1.0)) * (
                1 - smoothing_cdf_start(torch.remainder(foot_indices[2], 1.0) - 0.5)) +
                                    smoothing_cdf_start(torch.remainder(foot_indices[2], 1.0) - 1) * (
                                            1 - smoothing_cdf_start(
                                        torch.remainder(foot_indices[2], 1.0) - 0.5 - 1)))
        smoothing_multiplier_RR = (smoothing_cdf_start(torch.remainder(foot_indices[3], 1.0)) * (
                1 - smoothing_cdf_start(torch.remainder(foot_indices[3], 1.0) - 0.5)) +
                                    smoothing_cdf_start(torch.remainder(foot_indices[3], 1.0) - 1) * (
                                            1 - smoothing_cdf_start(
                                        torch.remainder(foot_indices[3], 1.0) - 0.5 - 1)))

        self.desired_contact_states[:, 0] = smoothing_multiplier_FL
        self.desired_contact_states[:, 1] = smoothing_multiplier_FR
        self.desired_contact_states[:, 2] = smoothing_multiplier_RL
        self.desired_contact_states[:, 3] = smoothing_multiplier_RR

        if self.cfg.commands.num_commands > 9:
            self.desired_footswing_height = self.commands[:, 9]

    def _compute_torques(self, actions):
        """ Compute torques from actions.
            Actions can be interpreted as position or velocity targets given to a PD controller, or directly as scaled torques.
            [NOTE]: torques must have the same dimension as the number of DOFs, even if some DOFs are not actuated.

        Args:
            actions (torch.Tensor): Actions

        Returns:
            [torch.Tensor]: Torques sent to the simulation
        """
        # pd controller
        actions_scaled = torch.zeros((actions.shape[0], self.num_dof)).to(self.device)
        actions_scaled[:, :self.num_actuated_dof] = actions[:, :self.num_actuated_dof] * self.cfg.control.action_scale
        if self.num_actions >= 12:
            actions_scaled[:, [0, 3, 6, 9]] *= self.cfg.control.hip_scale_reduction  # scale down hip flexion range

        self.lag_buffer = torch.cat((self.lag_buffer[1:], actions_scaled.clone().unsqueeze(0)), dim=0)
        # Lag buffer is a queue where actions closer to idx 0 are older
        # The index we retrieve is thus inverse of the lag timestep (higher lag means lower index)
        lag_timesteps_idx = self.lag_buffer.shape[0] - self.lag_timesteps - 1
        # Shape of lag buffer is (max lag timesteps, num envs, num actions), so we index into the first dimension
        # for the lag timestep, and into the second dimension for the corresponding env id
        lagged_actions = self.lag_buffer[lag_timesteps_idx, torch.arange(self.num_envs), :]
        self.joint_pos_target = lagged_actions + self.default_dof_pos

        control_type = self.cfg.control.control_type

        if control_type == "actuator_net":
            self.joint_pos_err = self.dof_pos - self.joint_pos_target + self.motor_offsets
            self.joint_vel = self.dof_vel
            torques = self.actuator_network(self.joint_pos_err, self.joint_pos_err_last, self.joint_pos_err_last_last,
                                            self.joint_vel, self.joint_vel_last, self.joint_vel_last_last)
            self.joint_pos_err_last_last = torch.clone(self.joint_pos_err_last)
            self.joint_pos_err_last = torch.clone(self.joint_pos_err)
            self.joint_vel_last_last = torch.clone(self.joint_vel_last)
            self.joint_vel_last = torch.clone(self.joint_vel)
        elif control_type == "P":
            torques = self.p_gains * self.Kp_factors * (
                    self.joint_pos_target - self.dof_pos + self.motor_offsets) - self.d_gains * self.Kd_factors * self.dof_vel
        else:
            raise NameError(f"Unknown controller type: {control_type}")

        torques = torques * self.motor_strengths
        return torch.clip(torques, -self.torque_limits, self.torque_limits)

    def _reset_dofs(self, env_ids, cfg):
        """ Resets DOF position and velocities of selected environmments
        Positions are randomly selected within 0.5:1.5 x default positions.
        Velocities are set to zero.

        Args:
            env_ids (List[int]): Environemnt ids
        """
        self.dof_pos[env_ids] = self.default_dof_pos * torch_rand_float(0.5, 1.5, (len(env_ids), self.num_dof), device=self.device)
        self.dof_vel[env_ids] = 0.

        all_subject_env_ids = self.robot_actor_idxs[env_ids].to(device=self.device)
        all_subject_env_ids_int32 = all_subject_env_ids.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(all_subject_env_ids_int32), len(all_subject_env_ids_int32))

    def _reset_root_states(self, env_ids, cfg):
        """ Resets ROOT states position and velocities of selected environmments
            Sets base position based on the curriculum
            Selects randomized base velocities within -0.5:0.5 [m/s, rad/s]
        Args:
            env_ids (List[int]): Environemnt ids
        """
        robot_env_ids = self.robot_actor_idxs[env_ids].to(device=self.device)

        ### Reset robots
        self.root_states[robot_env_ids] = self.base_init_state
        self.root_states[robot_env_ids, :3] += self.env_origins[env_ids]
        self.root_states[robot_env_ids, 0:1] += torch_rand_float(-cfg.terrain.x_init_range, cfg.terrain.x_init_range,
                                                                 (len(robot_env_ids), 1), device=self.device)
        self.root_states[robot_env_ids, 1:2] += torch_rand_float(-cfg.terrain.y_init_range, cfg.terrain.y_init_range,
                                                                 (len(robot_env_ids), 1), device=self.device)
        self.root_states[robot_env_ids, 0] += cfg.terrain.x_init_offset
        self.root_states[robot_env_ids, 1] += cfg.terrain.y_init_offset
        self.root_states[robot_env_ids, 2] += self.ball_radius[env_ids] * 2 + 0.0001

        random_yaw_angle = 2*(torch.rand(len(env_ids), 3, dtype=torch.float, device=self.device,
                                                     requires_grad=False)-0.5)*torch.tensor([0, 0, cfg.terrain.yaw_init_range], device=self.device)
        self.root_states[robot_env_ids,3:7] = quat_from_euler_xyz(random_yaw_angle[:,0], random_yaw_angle[:,1], random_yaw_angle[:,2])
            
        self.root_states[robot_env_ids, 7:13] = torch_rand_float(-0.5, 0.5, (len(robot_env_ids), 6), device=self.device)  # [7:10]: lin vel, [10:13]: ang vel

        ### Reset objects
        object_env_ids = self.object_actor_idxs[env_ids].to(device=self.device)

        self.root_states[object_env_ids] = self.object_init_state
        self.root_states[object_env_ids, :3] += self.env_origins[env_ids]
        self.root_states[object_env_ids,2] += self.ball_radius[env_ids] + 0.0001
        # self.root_states[object_env_ids,0:3] += 2*(torch.rand(len(env_ids), 3, dtype=torch.float, device=self.device,
        #                                             requires_grad=False)-0.5) * torch.tensor(cfg.ball.init_pos_range,device=self.device,
        #                                             requires_grad=False)
        # self.root_states[object_env_ids,7:10] += 2*(torch.rand(len(env_ids), 3, dtype=torch.float, device=self.device,
        #                                             requires_grad=False)-0.5) * torch.tensor(cfg.ball.init_vel_range,device=self.device,
        #                                             requires_grad=False)
        self.root_states[object_env_ids,0:3] += torch.tensor(cfg.ball.init_pos_range,device=self.device, requires_grad=False)
        self.root_states[object_env_ids,7:10] += torch.tensor(cfg.ball.init_vel_range,device=self.device, requires_grad=False)

        all_subject_env_ids = torch.cat((robot_env_ids, object_env_ids)) if self.cfg.env.add_balls else robot_env_ids
        all_subject_env_ids_int32 = all_subject_env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_states),
                                                     gymtorch.unwrap_tensor(all_subject_env_ids_int32), len(all_subject_env_ids_int32))

        if cfg.env.record_video and 0 in env_ids:
            if self.complete_video_frames is None:
                self.complete_video_frames = []
            else:
                self.complete_video_frames = self.video_frames[:]
            self.video_frames = []
        
    def _push_robots(self, env_ids, cfg):
        """ Random pushes the robots. Emulates an impulse by setting a randomized base velocity.
        """
        env_ids = env_ids[self.episode_length_buf[env_ids] % int(cfg.domain_rand.push_robot_interval) == 0]
        # print(env_ids)
        # print(self.episode_length_buf)
        # print(self.episode_length_buf[0])
        # print(env_ids.dtype)
        # print(self.episode_length_buf[env_ids])
        # print(int(cfg.domain_rand.push_robot_interval))
        min_push_vel, max_push_vel = cfg.domain_rand.robot_push_vel_range
        push_vel_mag = torch.rand(len(env_ids), dtype=torch.float, device=self.device, requires_grad=False) * (max_push_vel - min_push_vel) + min_push_vel
        push_vel_vec = torch_rand_float(-1, 1, (len(env_ids), 2), device=self.device)
        push_vel_vec = push_vel_vec / torch.norm(push_vel_vec, dim=1).unsqueeze(1)
        # print(push_vel_mag.unsqueeze(1) * push_vel_vec)
        self.root_states[self.robot_actor_idxs[env_ids], 7:9] += push_vel_mag.unsqueeze(1) * push_vel_vec  # lin vel x/y
        self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_states))

    def _push_balls(self, env_ids, cfg):
        """ Random pushes the balls. Emulates an impulse by setting a randomized base velocity.
        """
        env_ids = env_ids[self.episode_length_buf[env_ids] % int(cfg.domain_rand.push_ball_interval) == 0]
        min_push_vel, max_push_vel = cfg.domain_rand.ball_push_vel_range
        push_vel_mag = torch.rand(len(env_ids), dtype=torch.float, device=self.device, requires_grad=False) * (max_push_vel - min_push_vel) + min_push_vel
        push_vel_vec = torch_rand_float(-1, 1, (len(env_ids), 2), device=self.device)
        push_vel_vec = push_vel_vec / torch.norm(push_vel_vec, dim=1).unsqueeze(1)
        self.root_states[self.object_actor_idxs[env_ids], 7:9] += push_vel_mag.unsqueeze(1) * push_vel_vec  # lin vel x/y
        self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_states))
            
    def _update_terrain_curriculum(self, env_ids):
        """ Implements the game-inspired curriculum.
        Args:
            env_ids (List[int]): ids of environments being reset
        """
        # Implement Terrain curriculum
        robot_env_ids = self.robot_actor_idxs[env_ids]
        if not self.init_done:
            # don't change on initial reset
            return
        distance = torch.norm(self.root_states[robot_env_ids, :2] - self.env_origins[env_ids, :2], dim=1)
        # robots that walked far enough progress to harder terains
        move_up = distance > self.terrain.cfg.env_length / 2

        move_down = (self.path_distance[env_ids] < torch.norm(self.commands[env_ids, :2],
                                           dim=1) * self.max_episode_length_s * 0.5) * ~move_up
        self.terrain_levels[env_ids] += 1 * move_up - 1 * move_down
        # Robots that solve the last level are sent to a random xfone
        self.terrain_levels[env_ids] = torch.where(self.terrain_levels[env_ids] >= self.max_terrain_level,
                                                   torch.randint_like(self.terrain_levels[env_ids],
                                                                      low=self.min_terrain_level,
                                                                      high=self.max_terrain_level),
                                                   torch.clip(self.terrain_levels[env_ids],
                                                              self.min_terrain_level))  # (the minumum level is zero)
        self.env_origins[env_ids] = self.cfg.terrain.terrain_origins[self.terrain_levels[env_ids], self.terrain_types[env_ids]]
        
    def _update_command_ranges(self, env_ids):
        constrict_indices = self.cfg.rewards.constrict_indices
        constrict_ranges = self.cfg.rewards.constrict_ranges

        if self.cfg.rewards.constrict and self.common_step_counter >= self.cfg.rewards.constrict_after:
            for idx, range in zip(constrict_indices, constrict_ranges):
                self.commands[env_ids, idx] = range[0]

    def _teleport_robots(self, env_ids, cfg):
        """ Teleports any robots that are too close to the edge to the other side
        """
        robot_env_ids = self.robot_actor_idxs[env_ids].to(device=self.device)
        if cfg.terrain.teleport_robots:
            thresh = cfg.terrain.teleport_thresh

            x_offset = int(cfg.terrain.x_offset * cfg.terrain.horizontal_scale)

            low_x_ids = robot_env_ids[self.root_states[robot_env_ids, 0] < thresh + x_offset]
            self.root_states[low_x_ids, 0] += cfg.terrain.terrain_length * (cfg.terrain.num_rows - 1)

            high_x_ids = robot_env_ids[
                self.root_states[robot_env_ids, 0] > cfg.terrain.terrain_length * cfg.terrain.num_rows - thresh + x_offset]
            self.root_states[high_x_ids, 0] -= cfg.terrain.terrain_length * (cfg.terrain.num_rows - 1)

            low_y_ids = robot_env_ids[self.root_states[robot_env_ids, 1] < thresh]
            self.root_states[low_y_ids, 1] += cfg.terrain.terrain_width * (cfg.terrain.num_cols - 1)

            high_y_ids = robot_env_ids[
                self.root_states[robot_env_ids, 1] > cfg.terrain.terrain_width * cfg.terrain.num_cols - thresh]
            self.root_states[high_y_ids, 1] -= cfg.terrain.terrain_width * (cfg.terrain.num_cols - 1)

            self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_states))
            self.gym.refresh_actor_root_state_tensor(self.sim)

    # ----------------------------------------
    def _init_buffers(self):
        """ Initialize torch tensors which will contain simulation states and processed quantities
        """
        # get gym GPU state tensors
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)
        rigid_body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.render_all_camera_sensors(self.sim)

        # create some wrapper tensors for different slices
        self.root_states = gymtorch.wrap_tensor(actor_root_state)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.net_contact_forces = gymtorch.wrap_tensor(net_contact_forces)[:self.num_envs * self.num_bodies, :]
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]
        self.base_pos = self.root_states[self.robot_actor_idxs, 0:3]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]
        self.base_quat = self.root_states[self.robot_actor_idxs, 3:7]
        self.rigid_body_state = gymtorch.wrap_tensor(rigid_body_state)[:self.num_envs*self.total_rigid_body_num,:].view(self.num_envs,self.total_rigid_body_num,13)[:,0:self.num_bodies, :]#.contiguous().view(self.num_envs*self.num_bodies,13)
        self.rigid_body_state_object = gymtorch.wrap_tensor(rigid_body_state)[:self.num_envs*self.total_rigid_body_num,:].view(self.num_envs,self.total_rigid_body_num,13)[:,self.num_bodies:self.num_bodies + self.num_object_bodies, :]#.contiguous().view(self.num_envs*self.num_bodies,13)
        self.foot_velocities = self.rigid_body_state.view(self.num_envs, -1, 13)[:,
                               self.feet_indices,
                               7:10]
        self.foot_positions = self.rigid_body_state.view(self.num_envs, -1, 13)[:, self.feet_indices,
                              0:3]
        self.prev_base_pos = self.base_pos.clone()
        self.prev_foot_velocities = self.foot_velocities.clone()

        _, max_lag_timesteps = self.cfg.domain_rand.lag_timesteps_range
        self.lag_buffer = torch.zeros((int(max_lag_timesteps)+1, *self.dof_pos.shape), device=self.device)

        self.contact_forces = gymtorch.wrap_tensor(net_contact_forces)[:self.num_envs*self.total_rigid_body_num,:].view(self.num_envs,self.total_rigid_body_num,3)[:,0:self.num_bodies, :]
        self.last_contact_forces = torch.zeros_like(self.contact_forces)

        self.object_pos_world_frame = self.root_states[self.object_actor_idxs, 0:3]
        robot_object_vec = self.asset.get_local_pos()
        self.object_local_pos = quat_rotate_inverse(self.base_quat, robot_object_vec)
        self.object_local_pos[:, 2] = 0.0*torch.ones(self.num_envs, dtype=torch.float,
                                    device=self.device, requires_grad=False)

        self.last_object_local_pos = torch.clone(self.object_local_pos)
        self.object_lin_vel = self.asset.get_lin_vel()
        self.object_ang_vel = self.asset.get_ang_vel()
         

        # initialize some data used later on
        self.common_step_counter = 0
        self.extras = {}

        self.measured_heights = 0

        self.gravity_vec = to_torch(get_axis_params(-1., self.up_axis_idx), device=self.device).repeat(
            (self.num_envs, 1))
        self.forward_vec = to_torch([1., 0., 0.], device=self.device).repeat((self.num_envs, 1))
        self.torques = torch.zeros(self.num_envs, self.num_dof, dtype=torch.float, device=self.device,
                                   requires_grad=False)
        self.forces = torch.zeros(self.num_envs, self.total_rigid_body_num, 3, dtype=torch.float, device=self.device,
                                   requires_grad=False)
        self.p_gains = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        self.d_gains = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        self.actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device,
                                   requires_grad=False)
        self.last_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device,
                                        requires_grad=False)
        self.last_last_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device,
                                             requires_grad=False)
        self.joint_pos_target = torch.zeros(self.num_envs, self.num_dof, dtype=torch.float,
                                            device=self.device,
                                            requires_grad=False)
        self.last_joint_pos_target = torch.zeros(self.num_envs, self.num_dof, dtype=torch.float, device=self.device,
                                                 requires_grad=False)
        self.last_last_joint_pos_target = torch.zeros(self.num_envs, self.num_dof, dtype=torch.float,
                                                      device=self.device,
                                                      requires_grad=False)
        self.last_dof_vel = torch.zeros_like(self.dof_vel)
        self.last_root_vel = torch.zeros_like(self.root_states[self.robot_actor_idxs, 7:13])
        self.path_distance = torch.zeros(self.num_envs, dtype=torch.float, device=self.device,
                                                 requires_grad=False)
        self.past_base_pos = self.base_pos.clone()


        self.commands_value = torch.zeros(self.num_envs, self.cfg.commands.num_commands, dtype=torch.float,
                                          device=self.device, requires_grad=False)
        self.commands = torch.zeros_like(self.commands_value)  # x vel, y vel, yaw vel, heading
        self.heading_commands = torch.zeros(self.num_envs, dtype=torch.float,
                                          device=self.device, requires_grad=False)  # heading
        self.heading_offsets = torch.zeros(self.num_envs, dtype=torch.float,
                                            device=self.device, requires_grad=False)  # heading offset
        self.commands_scale = torch.tensor([self.obs_scales.lin_vel, self.obs_scales.lin_vel, self.obs_scales.ang_vel,
                                            self.obs_scales.body_height_cmd, self.obs_scales.gait_freq_cmd,
                                            self.obs_scales.gait_phase_cmd, self.obs_scales.gait_phase_cmd,
                                            self.obs_scales.gait_phase_cmd, self.obs_scales.gait_phase_cmd,
                                            self.obs_scales.footswing_height_cmd, self.obs_scales.body_pitch_cmd,
                                            self.obs_scales.body_roll_cmd, self.obs_scales.stance_width_cmd,
                                           self.obs_scales.stance_length_cmd, self.obs_scales.aux_reward_cmd],
                                           device=self.device, requires_grad=False, )[:self.cfg.commands.num_commands]
        self.desired_contact_states = torch.zeros(self.num_envs, 4, dtype=torch.float, device=self.device,
                                                  requires_grad=False, )


        self.feet_air_time = torch.zeros(self.num_envs, self.feet_indices.shape[0], dtype=torch.float,
                                         device=self.device, requires_grad=False)
        self.last_contacts = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.bool, device=self.device,
                                         requires_grad=False)
        self.last_contact_filt = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.bool,
                                             device=self.device,
                                             requires_grad=False)
        self.base_lin_vel = quat_rotate_inverse(self.base_quat, self.root_states[self.robot_actor_idxs, 7:10])
        self.base_ang_vel = quat_rotate_inverse(self.base_quat, self.root_states[self.robot_actor_idxs, 10:13])
       
        
        self.projected_gravity = quat_rotate_inverse(self.base_quat, self.gravity_vec)

        # joint positions offsets and PD gains
        self.default_dof_pos = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        for i in range(self.num_dofs):
            name = self.dof_names[i]
            angle = self.cfg.init_state.default_joint_angles[name]
            self.default_dof_pos[i] = angle
            found = False
            for dof_name in self.cfg.control.stiffness.keys():
                if dof_name in name:
                    self.p_gains[i] = self.cfg.control.stiffness[dof_name]
                    self.d_gains[i] = self.cfg.control.damping[dof_name]
                    found = True
            if not found:
                self.p_gains[i] = 0.
                self.d_gains[i] = 0.
                if self.cfg.control.control_type in ["P", "V"]:
                    print(f"PD gain of joint {name} were not defined, setting them to zero")
        self.default_dof_pos = self.default_dof_pos.unsqueeze(0)

        if self.cfg.control.control_type == "actuator_net":
            actuator_path = f'{os.path.dirname(os.path.dirname(os.path.realpath(__file__)))}/../../resources/actuator_nets/unitree_go1.pt'
            actuator_network = torch.jit.load(actuator_path, map_location=self.device)

            def eval_actuator_network(joint_pos, joint_pos_last, joint_pos_last_last, joint_vel, joint_vel_last,
                                      joint_vel_last_last):
                xs = torch.cat((joint_pos.unsqueeze(-1),
                                joint_pos_last.unsqueeze(-1),
                                joint_pos_last_last.unsqueeze(-1),
                                joint_vel.unsqueeze(-1),
                                joint_vel_last.unsqueeze(-1),
                                joint_vel_last_last.unsqueeze(-1)), dim=-1)
                torques = actuator_network(xs.view(self.num_envs * 12, 6))
                return torques.view(self.num_envs, 12)

            self.actuator_network = eval_actuator_network

            self.joint_pos_err_last_last = torch.zeros((self.num_envs, 12), device=self.device)
            self.joint_pos_err_last = torch.zeros((self.num_envs, 12), device=self.device)
            self.joint_vel_last_last = torch.zeros((self.num_envs, 12), device=self.device)
            self.joint_vel_last = torch.zeros((self.num_envs, 12), device=self.device)

    def _init_custom_buffers__(self):
        # domain randomization properties
        self.robot_friction_coeffs = self.default_robot_friction * torch.ones(self.num_envs, 4, dtype=torch.float, device=self.device,
                                                                  requires_grad=False)
        self.robot_restitutions = self.default_robot_restitution * torch.ones(self.num_envs, 4, dtype=torch.float, device=self.device,
                                                                  requires_grad=False)
        self.payloads = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self.com_displacements = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device,
                                             requires_grad=False)
        self.motor_strengths = torch.ones(self.num_envs, self.num_dof, dtype=torch.float, device=self.device,
                                          requires_grad=False)
        self.motor_offsets = torch.zeros(self.num_envs, self.num_dof, dtype=torch.float, device=self.device,
                                         requires_grad=False)
        self.Kp_factors = torch.ones(self.num_envs, self.num_dof, dtype=torch.float, device=self.device,
                                     requires_grad=False)
        self.Kd_factors = torch.ones(self.num_envs, self.num_dof, dtype=torch.float, device=self.device,
                                     requires_grad=False)

        self.ball_radius = self.default_ball_radius * torch.ones(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self.ball_masses = self.default_ball_mass * torch.ones(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self.ball_friction_coeffs = self.default_ball_friction * torch.ones(self.num_envs, dtype=torch.float,
                                                                  device=self.device,
                                                                  requires_grad=False)
        self.ball_restitutions = self.default_ball_restitution * torch.ones(self.num_envs, dtype=torch.float, device=self.device,
                                                                  requires_grad=False)
        self.ball_compliances = self.default_ball_compliance * torch.ones(self.num_envs, dtype=torch.float, device=self.device,
                                                                  requires_grad=False)
        self.ball_inertia_multipliers = self.default_ball_inertia_multiplier * torch.ones(self.num_envs, dtype=torch.float, device=self.device,
                                                                    requires_grad=False)
        self.ball_spring_coefficients = self.default_ball_spring_coefficient * torch.ones(self.num_envs, dtype=torch.float, device=self.device,
                                                                    requires_grad=False)
        self.ball_drags = torch.zeros(self.num_envs, 1, dtype=torch.float, device=self.device,
                                     requires_grad=False)

        self.gravities = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device,
                                     requires_grad=False)
        self.gravity_vec = to_torch(get_axis_params(-1., self.up_axis_idx), device=self.device).repeat(
            (self.num_envs, 1))
        
        self.lag_timesteps = self.default_lag_timesteps * torch.ones(self.num_envs, dtype=torch.long, device=self.device,
                                                                        requires_grad=False)

        # if custom initialization values were passed in, set them here
        dynamics_params = ["friction_coeffs", "restitutions", "payloads", "com_displacements", "motor_strengths",
                           "Kp_factors", "Kd_factors"]
        if self.initial_dynamics_dict is not None:
            print("WARNING: using custom dynamics initialization (self.initial_dynamics_dict)")
            for k, v in self.initial_dynamics_dict.items():
                if k in dynamics_params:
                    setattr(self, k, v.to(self.device))

        self.gait_indices = torch.zeros(self.num_envs, dtype=torch.float, device=self.device,
                                        requires_grad=False)
        self.clock_inputs = torch.zeros(self.num_envs, 4, dtype=torch.float, device=self.device,
                                        requires_grad=False)
        self.doubletime_clock_inputs = torch.zeros(self.num_envs, 4, dtype=torch.float, device=self.device,
                                                   requires_grad=False)
        self.halftime_clock_inputs = torch.zeros(self.num_envs, 4, dtype=torch.float, device=self.device,
                                                 requires_grad=False)

    def _init_command_distribution(self, env_ids):
        # new style curriculum
        self.category_names = ['nominal']
        if self.cfg.commands.gaitwise_curricula:
            self.category_names = ['pronk', 'trot', 'pace', 'bound']

        if self.cfg.commands.curriculum_type == "RewardThresholdCurriculum":
            from .curriculum import RewardThresholdCurriculum
            CurriculumClass = RewardThresholdCurriculum
        self.curricula = []
        for category in self.category_names:
            self.curricula += [CurriculumClass(seed=self.cfg.commands.curriculum_seed,
                                               x_vel=(self.cfg.commands.limit_vel_x[0],
                                                      self.cfg.commands.limit_vel_x[1],
                                                      self.cfg.commands.num_bins_vel_x),
                                               y_vel=(self.cfg.commands.limit_vel_y[0],
                                                      self.cfg.commands.limit_vel_y[1],
                                                      self.cfg.commands.num_bins_vel_y),
                                               yaw_vel=(self.cfg.commands.limit_vel_yaw[0],
                                                        self.cfg.commands.limit_vel_yaw[1],
                                                        self.cfg.commands.num_bins_vel_yaw),
                                               body_height=(self.cfg.commands.limit_body_height[0],
                                                            self.cfg.commands.limit_body_height[1],
                                                            self.cfg.commands.num_bins_body_height),
                                               gait_frequency=(self.cfg.commands.limit_gait_frequency[0],
                                                               self.cfg.commands.limit_gait_frequency[1],
                                                               self.cfg.commands.num_bins_gait_frequency),
                                               gait_phase=(self.cfg.commands.limit_gait_phase[0],
                                                           self.cfg.commands.limit_gait_phase[1],
                                                           self.cfg.commands.num_bins_gait_phase),
                                               gait_offset=(self.cfg.commands.limit_gait_offset[0],
                                                            self.cfg.commands.limit_gait_offset[1],
                                                            self.cfg.commands.num_bins_gait_offset),
                                               gait_bounds=(self.cfg.commands.limit_gait_bound[0],
                                                            self.cfg.commands.limit_gait_bound[1],
                                                            self.cfg.commands.num_bins_gait_bound),
                                               gait_duration=(self.cfg.commands.limit_gait_duration[0],
                                                              self.cfg.commands.limit_gait_duration[1],
                                                              self.cfg.commands.num_bins_gait_duration),
                                               footswing_height=(self.cfg.commands.limit_footswing_height[0],
                                                                 self.cfg.commands.limit_footswing_height[1],
                                                                 self.cfg.commands.num_bins_footswing_height),
                                               body_pitch=(self.cfg.commands.limit_body_pitch[0],
                                                           self.cfg.commands.limit_body_pitch[1],
                                                           self.cfg.commands.num_bins_body_pitch),
                                               body_roll=(self.cfg.commands.limit_body_roll[0],
                                                          self.cfg.commands.limit_body_roll[1],
                                                          self.cfg.commands.num_bins_body_roll),
                                               stance_width=(self.cfg.commands.limit_stance_width[0],
                                                             self.cfg.commands.limit_stance_width[1],
                                                             self.cfg.commands.num_bins_stance_width),
                                               stance_length=(self.cfg.commands.limit_stance_length[0],
                                                                self.cfg.commands.limit_stance_length[1],
                                                                self.cfg.commands.num_bins_stance_length),
                                               aux_reward_coef=(self.cfg.commands.limit_aux_reward_coef[0],
                                                           self.cfg.commands.limit_aux_reward_coef[1],
                                                           self.cfg.commands.num_bins_aux_reward_coef),
                                               )]

        if self.cfg.commands.curriculum_type == "LipschitzCurriculum":
            for curriculum in self.curricula:
                curriculum.set_params(lipschitz_threshold=self.cfg.commands.lipschitz_threshold,
                                      binary_phases=self.cfg.commands.binary_phases)
        self.env_command_bins = np.zeros(len(env_ids), dtype=np.int)
        self.env_command_categories = np.zeros(len(env_ids), dtype=np.int)
        low = np.array(
            [self.cfg.commands.lin_vel_x[0], self.cfg.commands.lin_vel_y[0],
             self.cfg.commands.ang_vel_yaw[0], self.cfg.commands.body_height_cmd[0],
             self.cfg.commands.gait_frequency_cmd_range[0],
             self.cfg.commands.gait_phase_cmd_range[0], self.cfg.commands.gait_offset_cmd_range[0],
             self.cfg.commands.gait_bound_cmd_range[0], self.cfg.commands.gait_duration_cmd_range[0],
             self.cfg.commands.footswing_height_range[0], self.cfg.commands.body_pitch_range[0],
             self.cfg.commands.body_roll_range[0],self.cfg.commands.stance_width_range[0],
             self.cfg.commands.stance_length_range[0], self.cfg.commands.aux_reward_coef_range[0], ])
        high = np.array(
            [self.cfg.commands.lin_vel_x[1], self.cfg.commands.lin_vel_y[1],
             self.cfg.commands.ang_vel_yaw[1], self.cfg.commands.body_height_cmd[1],
             self.cfg.commands.gait_frequency_cmd_range[1],
             self.cfg.commands.gait_phase_cmd_range[1], self.cfg.commands.gait_offset_cmd_range[1],
             self.cfg.commands.gait_bound_cmd_range[1], self.cfg.commands.gait_duration_cmd_range[1],
             self.cfg.commands.footswing_height_range[1], self.cfg.commands.body_pitch_range[1],
             self.cfg.commands.body_roll_range[1],self.cfg.commands.stance_width_range[1],
             self.cfg.commands.stance_length_range[1], self.cfg.commands.aux_reward_coef_range[1], ])
        for curriculum in self.curricula:
            curriculum.set_to(low=low, high=high)

    def _prepare_reward_function(self):
        """ Prepares a list of reward functions, whcih will be called to compute the total reward.
            Looks for self._reward_<REWARD_NAME>, where <REWARD_NAME> are names of all non zero reward scales in the cfg.
        """
        # reward containers
        from globe_walking.go1_gym.rewards.eureka_reward import EurekaReward
        reward_containers = {"EurekaReward": EurekaReward}
        self.reward_container = reward_containers[self.cfg.rewards.reward_container_name](self)

        if "compute_reward" in dir(self.reward_container):
            exit()
            _, reward_components = self.reward_container.compute_reward()
            self.reward_names = list(reward_components.keys())
            self.reward_functions = None
        else:
            # prepare list of functions
            self.reward_functions = []
            self.reward_names = []
            for name in dir(self.reward_container):
                if not name.startswith("_reward_"):
                    continue
                name = name.replace("_reward_", "")
                self.reward_names.append(name)
                self.reward_functions.append(getattr(self.reward_container, '_reward_' + name))

        # reward episode sums
        self.episode_sums = {
            name: torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
            for name in self.reward_names}
        self.episode_sums["total"] = torch.zeros(self.num_envs, dtype=torch.float, device=self.device,
                                                 requires_grad=False)
        self.episode_sums["success"] = torch.zeros(self.num_envs, dtype=torch.float, device=self.device,
                                                 requires_grad=False)
        self.command_sums = {
            name: torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
            for name in
            list(self.reward_names) + ["lin_vel_raw", "ang_vel_raw", "lin_vel_residual", "ang_vel_residual",
                                               "ep_timesteps"]}

    def _create_envs(self):
        self.default_robot_friction = (self.cfg.domain_rand.robot_friction_range[0] + self.cfg.domain_rand.robot_friction_range[1]) / 2
        self.default_robot_restitution = (self.cfg.domain_rand.robot_restitution_range[0] + self.cfg.domain_rand.robot_restitution_range[1]) / 2
        self.default_ball_radius = (self.cfg.domain_rand.ball_radius_range[0] + self.cfg.domain_rand.ball_radius_range[1]) / 2
        self.default_ball_mass = (self.cfg.domain_rand.ball_mass_range[0] + self.cfg.domain_rand.ball_mass_range[1]) / 2
        self.default_ball_friction = (self.cfg.domain_rand.ball_friction_range[0] + self.cfg.domain_rand.ball_friction_range[1]) / 2
        self.default_ball_restitution = (self.cfg.domain_rand.ball_restitution_range[0] + self.cfg.domain_rand.ball_restitution_range[1]) / 2
        self.default_ball_compliance = (self.cfg.domain_rand.ball_compliance_range[0] + self.cfg.domain_rand.ball_compliance_range[1]) / 2
        self.default_ball_inertia_multiplier = (self.cfg.domain_rand.ball_inertia_multiplier_range[0] + self.cfg.domain_rand.ball_inertia_multiplier_range[1]) / 2
        self.default_ball_spring_coefficient = (self.cfg.domain_rand.ball_spring_coefficient_range[0] + self.cfg.domain_rand.ball_spring_coefficient_range[1]) / 2
        self.default_lag_timesteps = int((self.cfg.domain_rand.lag_timesteps_range[0] + self.cfg.domain_rand.lag_timesteps_range[1]) / 2)

        all_assets = []

        # create robot
        from globe_walking.go1_gym.robots.go1 import Go1

        robot_classes = {
            'go1': Go1,
        }

        self.robot = robot_classes[self.cfg.robot.name](self)
        all_assets.append(self.robot)
        self.robot_asset, dof_props_asset, rigid_shape_props_asset = self.robot.initialize()
        

        object_init_state_list = self.cfg.ball.ball_init_pos + self.cfg.ball.ball_init_rot + self.cfg.ball.ball_init_lin_vel + self.cfg.ball.ball_init_ang_vel
        self.object_init_state = to_torch(object_init_state_list, device=self.device, requires_grad=False)

        # create objects
        from globe_walking.go1_gym.assets.ball import Ball

        asset_classes = {
            "ball": Ball,
        }

        self.asset = asset_classes[self.cfg.ball.asset](self, self.default_ball_radius)
        all_assets.append(self.asset)
        self.ball_asset, ball_rigid_shape_props_asset = self.asset.initialize()
        self.ball_force_feedback = self.asset.get_force_feedback()
        self.num_object_bodies = self.gym.get_asset_rigid_body_count(self.ball_asset)

        # aggregate the asset properties
        self.total_rigid_body_num = sum([asset.get_num_bodies() for asset in 
                                        all_assets])
        self.num_dof = sum([asset.get_num_dof() for asset in
                            all_assets])
        self.num_actuated_dof = sum([asset.get_num_actuated_dof() for asset in
                                        all_assets])
        self.dof_names = self.gym.get_asset_dof_names(self.robot_asset)

        if self.cfg.terrain.mesh_type == "boxes":
            self.total_rigid_body_num += self.cfg.terrain.num_cols * self.cfg.terrain.num_rows

        

        self.ball_init_pose = gymapi.Transform()
        self.ball_init_pose.p = gymapi.Vec3(*self.object_init_state[:3])

        # save body names from the asset
        body_names = self.gym.get_asset_rigid_body_names(self.robot_asset)
        self.dof_names = self.gym.get_asset_dof_names(self.robot_asset)
        self.num_bodies = len(body_names)
        self.num_dofs = len(self.dof_names)
        feet_names = [s for s in body_names if self.cfg.asset.foot_name in s]
        penalized_contact_names = []
        for name in self.cfg.asset.penalize_contacts_on:
            penalized_contact_names.extend([s for s in body_names if name in s])
        termination_contact_names = []
        for name in self.cfg.asset.terminate_after_contacts_on:
            termination_contact_names.extend([s for s in body_names if name in s])

        base_init_state_list = self.cfg.init_state.pos + self.cfg.init_state.rot + self.cfg.init_state.lin_vel + self.cfg.init_state.ang_vel
        self.base_init_state = to_torch(base_init_state_list, device=self.device, requires_grad=False)
        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*self.base_init_state[:3])

        self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
        self.terrain_levels = torch.zeros(self.num_envs, device=self.device, requires_grad=False, dtype=torch.long)
        self.terrain_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
        self.terrain_types = torch.zeros(self.num_envs, device=self.device, requires_grad=False, dtype=torch.long)
        self._get_env_origins(torch.arange(self.num_envs, device=self.device), self.cfg)
        env_lower = gymapi.Vec3(0., 0., 0.)
        env_upper = gymapi.Vec3(0., 0., 0.)
        self.robot_actor_handles = []
        self.object_actor_handles = []
        self.imu_sensor_handles = []
        self.envs = []
        self.robot_actor_idxs = []
        self.object_actor_idxs = []

        self.object_rigid_body_idxs = []
        self.feet_rigid_body_idxs = []
        self.robot_rigid_body_idxs = []

        self._init_custom_buffers__()
        self._randomize_rigid_body_props(torch.arange(self.num_envs, device=self.device), self.cfg)
        self._randomize_gravity()
        self._randomize_ball_drag()

        for i in range(self.num_envs):
            # create env instance
            env_handle = self.gym.create_env(self.sim, env_lower, env_upper, int(np.sqrt(self.num_envs)))
            pos = self.env_origins[i].clone()
            # @willjhliang randomize initial position
            pos[0:1] += torch_rand_float(-self.cfg.terrain.x_init_range, self.cfg.terrain.x_init_range, (1, 1),
                                         device=self.device).squeeze(1)
            pos[1:2] += torch_rand_float(-self.cfg.terrain.y_init_range, self.cfg.terrain.y_init_range, (1, 1),
                                         device=self.device).squeeze(1)
            start_pose.p = gymapi.Vec3(*pos)

            # add robots
            rigid_shape_props = self._process_rigid_shape_props(rigid_shape_props_asset, i)
            self.gym.set_asset_rigid_shape_properties(self.robot_asset, rigid_shape_props)
            robot_handle = self.gym.create_actor(env_handle, self.robot_asset, start_pose, "robot", i,
                                                  self.cfg.asset.self_collisions, 0)
            for bi in body_names:
                self.robot_rigid_body_idxs.append(self.gym.find_actor_rigid_body_handle(env_handle, robot_handle, bi))
            dof_props = self._process_dof_props(dof_props_asset, i)
            self.gym.set_actor_dof_properties(env_handle, robot_handle, dof_props)
            body_props = self.gym.get_actor_rigid_body_properties(env_handle, robot_handle)
            body_props = self._process_robot_rigid_body_props(body_props, i)
            self.gym.set_actor_rigid_body_properties(env_handle, robot_handle, body_props, recomputeInertia=True)
            
            self.robot_actor_handles.append(robot_handle)
            self.robot_actor_idxs.append(self.gym.get_actor_index(env_handle, robot_handle, gymapi.DOMAIN_SIM))

            # add objects
            ball_rigid_shape_props = self._process_ball_rigid_shape_props(ball_rigid_shape_props_asset, i)
            self.gym.set_asset_rigid_shape_properties(self.ball_asset, ball_rigid_shape_props)
            ball_handle = self.gym.create_actor(env_handle, self.ball_asset, self.ball_init_pose, "ball", i, 0)
            color = gymapi.Vec3(0.25, 0.5, 1)
            self.gym.set_rigid_body_color(env_handle, ball_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, color)
            ball_idx = self.gym.get_actor_rigid_body_index(env_handle, ball_handle, 0, gymapi.DOMAIN_SIM)
            self.gym.set_actor_scale(env_handle, ball_handle, self.ball_radius[i] / self.default_ball_radius)
            ball_body_props = self.gym.get_actor_rigid_body_properties(env_handle, ball_handle)
            ball_body_props = self._process_ball_rigid_body_props(ball_body_props, i)
            self.gym.set_actor_rigid_body_properties(env_handle, ball_handle, ball_body_props, recomputeInertia=True)
            self.object_actor_handles.append(ball_handle)
            self.object_rigid_body_idxs.append(ball_idx)
            self.object_actor_idxs.append(self.gym.get_actor_index(env_handle, ball_handle, gymapi.DOMAIN_SIM))
                

            self.envs.append(env_handle)

        self.robot_actor_idxs = torch.Tensor(self.robot_actor_idxs).to(device=self.device,dtype=torch.long)
        self.object_actor_idxs = torch.Tensor(self.object_actor_idxs).to(device=self.device,dtype=torch.long)
        self.object_rigid_body_idxs = torch.Tensor(self.object_rigid_body_idxs).to(device=self.device,dtype=torch.long)
        
            
        self.feet_indices = torch.zeros(len(feet_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(feet_names)):
            self.feet_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.robot_actor_handles[0],
                                                                         feet_names[i])

        self.penalised_contact_indices = torch.zeros(len(penalized_contact_names), dtype=torch.long, device=self.device,
                                                     requires_grad=False)
        for i in range(len(penalized_contact_names)):
            self.penalised_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0],
                                                                                      self.robot_actor_handles[0],
                                                                                      penalized_contact_names[i])

        self.termination_contact_indices = torch.zeros(len(termination_contact_names), dtype=torch.long,
                                                       device=self.device, requires_grad=False)
        for i in range(len(termination_contact_names)):
            self.termination_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0],
                                                                                        self.robot_actor_handles[0],
                                                                                        termination_contact_names[i])
        ################
        ### Add sensors
        ################

        self.initialize_sensors()
        
        # if perception is on, set up camera
        if self.cfg.perception.compute_segmentation or self.cfg.perception.compute_rgb or self.cfg.perception.compute_depth:
            self.initialize_cameras(range(self.num_envs))

        if self.cfg.perception.measure_heights:
            from globe_walking.go1_gym.sensors.heightmap_sensor import HeightmapSensor
            self.heightmap_sensor = HeightmapSensor(self)

        # if recording video, set up camera
        if self.cfg.env.record_video:
            from globe_walking.go1_gym.sensors.floating_camera_sensor import FloatingCameraSensor
            self.rendering_camera = FloatingCameraSensor(self)
            

        ################
        ### Initialize Logging
        ################

        from globe_walking.go1_gym.utils.logger import Logger
        self.logger = Logger(self)
        
        self.video_writer = None
        self.video_frames = []
        self.complete_video_frames = []

    def render(self, mode="rgb_array", target_loc=None, cam_distance=None):
        self.rendering_camera.set_position(target_loc, cam_distance)
        return self.rendering_camera.get_observation()

    def _render_headless(self):
        if self.record_now and self.complete_video_frames is not None and len(self.complete_video_frames) == 0:
            bx, by, bz = self.root_states[self.robot_actor_idxs[0], 0], self.root_states[self.robot_actor_idxs[0], 1], self.root_states[self.robot_actor_idxs[0], 2]
            target_loc = [bx, by , bz]
            cam_distance = [0, -1.0, 1.0]
            self.rendering_camera.set_position(target_loc, cam_distance)
            self.video_frame = self.rendering_camera.get_observation()
            self.video_frames.append(self.video_frame)

    def start_recording(self):
        self.complete_video_frames = None
        self.record_now = True

    def pause_recording(self):
        self.complete_video_frames = []
        self.video_frames = []
        self.record_now = False

    def get_complete_frames(self):
        if self.complete_video_frames is None:
            return []
        return self.complete_video_frames

    def _get_env_origins(self, env_ids, cfg):
        """ Sets environment origins. On rough terrain the origins are defined by the terrain platforms.
            Otherwise create a grid.
        """
        if cfg.terrain.mesh_type in ["heightfield", "trimesh"]:
            self.custom_origins = True
            # put robots at the origins defined by the terrain
            max_init_level = cfg.terrain.max_init_terrain_level
            min_init_level = cfg.terrain.min_init_terrain_level
            if not cfg.terrain.curriculum: max_init_level = cfg.terrain.num_rows - 1
            if not cfg.terrain.curriculum: min_init_level = 0
            if cfg.terrain.center_robots:
                min_terrain_level = cfg.terrain.num_rows // 2 - cfg.terrain.center_span
                max_terrain_level = cfg.terrain.num_rows // 2 + cfg.terrain.center_span - 1
                min_terrain_type = cfg.terrain.num_cols // 2 - cfg.terrain.center_span
                max_terrain_type = cfg.terrain.num_cols // 2 + cfg.terrain.center_span - 1
                self.terrain_levels[env_ids] = torch.randint(min_terrain_level, max_terrain_level + 1, (len(env_ids),),
                                                             device=self.device)
                self.terrain_types[env_ids] = torch.randint(min_terrain_type, max_terrain_type + 1, (len(env_ids),),
                                                            device=self.device)
            else:
                self.terrain_levels[env_ids] = torch.randint(min_init_level, max_init_level + 1, (len(env_ids),),
                                                            device=self.device)
                self.terrain_types[env_ids] = torch.div(torch.arange(len(env_ids), device=self.device),
                                                    (len(env_ids) / cfg.terrain.num_cols), rounding_mode='floor').to(
                    torch.long)
            cfg.terrain.max_terrain_level = cfg.terrain.num_rows
            cfg.terrain.terrain_origins = torch.from_numpy(cfg.terrain.env_origins).to(self.device).to(torch.float)
            self.env_origins[env_ids] = cfg.terrain.terrain_origins[
                self.terrain_levels[env_ids], self.terrain_types[env_ids]]
        elif cfg.terrain.mesh_type in ["boxes", "boxes_tm"]:
            self.custom_origins = True
            # put robots at the origins defined by the terrain
            max_init_level = int(cfg.terrain.max_init_terrain_level + cfg.terrain.num_border_boxes)
            min_init_level = int(cfg.terrain.min_init_terrain_level + cfg.terrain.num_border_boxes)
            if not cfg.terrain.curriculum: max_init_level = int(cfg.terrain.num_rows - 1 - cfg.terrain.num_border_boxes)
            if not cfg.terrain.curriculum: min_init_level = int(0 + cfg.terrain.num_border_boxes)

            if cfg.terrain.center_robots:
                self.min_terrain_level = cfg.terrain.num_rows // 2 - cfg.terrain.center_span
                self.max_terrain_level = cfg.terrain.num_rows // 2 + cfg.terrain.center_span - 1
                min_terrain_type = cfg.terrain.num_cols // 2 - cfg.terrain.center_span
                max_terrain_type = cfg.terrain.num_cols // 2 + cfg.terrain.center_span - 1
                self.terrain_levels[env_ids] = torch.randint(self.min_terrain_level, self.max_terrain_level + 1, (len(env_ids),),
                                                             device=self.device)
                self.terrain_types[env_ids] = torch.randint(min_terrain_type, max_terrain_type + 1, (len(env_ids),),
                                                            device=self.device)
            else:
                self.terrain_levels[env_ids] = torch.randint(min_init_level, max_init_level + 1, (len(env_ids),),
                                                             device=self.device)
                self.terrain_types[env_ids] = (torch.div(torch.arange(len(env_ids), device=self.device),
                                                        (len(env_ids) / (cfg.terrain.num_cols - 2 * cfg.terrain.num_border_boxes)),
                                                        rounding_mode='floor') + cfg.terrain.num_border_boxes).to(torch.long)
                self.min_terrain_level = int(cfg.terrain.num_border_boxes)
                self.max_terrain_level = int(cfg.terrain.num_rows - cfg.terrain.num_border_boxes)
            cfg.terrain.env_origins[:, :, 2] = self.terrain_obj.terrain_cell_center_heights.cpu().numpy()
            cfg.terrain.terrain_origins = torch.from_numpy(cfg.terrain.env_origins).to(self.device).to(torch.float)
            self.env_origins[env_ids] = cfg.terrain.terrain_origins[self.terrain_levels[env_ids], self.terrain_types[env_ids]]
        else:
            self.custom_origins = False
            # create a grid of robots
            num_cols = np.floor(np.sqrt(len(env_ids)))
            num_rows = np.ceil(self.num_envs / num_cols)
            xx, yy = torch.meshgrid(torch.arange(num_rows), torch.arange(num_cols))
            spacing = cfg.env.env_spacing
            self.env_origins[env_ids, 0] = spacing * xx.flatten()[:len(env_ids)]
            self.env_origins[env_ids, 1] = spacing * yy.flatten()[:len(env_ids)]
            self.env_origins[env_ids, 2] = 0.

    def _parse_cfg(self, cfg):
        self.dt = self.cfg.control.decimation * self.sim_params.dt
        self.obs_scales = self.cfg.obs_scales
        self.curriculum_thresholds = vars(self.cfg.curriculum_thresholds)
        cfg.command_ranges = vars(cfg.commands)
        if cfg.terrain.mesh_type not in ['heightfield', 'trimesh', 'boxes', 'boxes_tm']:
            cfg.terrain.curriculum = False
        self.max_episode_length_s = cfg.env.episode_length_s
        cfg.env.max_episode_length = np.ceil(self.max_episode_length_s / self.dt)
        self.max_episode_length = cfg.env.max_episode_length

        cfg.domain_rand.push_robot_interval = np.ceil(cfg.domain_rand.push_robot_interval_s / self.dt)
        cfg.domain_rand.push_ball_interval = np.ceil(cfg.domain_rand.push_ball_interval_s / self.dt)
        # cfg.domain_rand.rand_interval = np.ceil(cfg.domain_rand.rand_interval_s / self.dt)
        cfg.domain_rand.gravity_rand_interval = np.ceil(cfg.domain_rand.gravity_rand_interval_s / self.dt)
        cfg.domain_rand.ball_drag_rand_interval = np.ceil(cfg.domain_rand.ball_drag_rand_interval_s / self.dt)
        # cfg.domain_rand.gravity_rand_duration = np.ceil(
        #     cfg.domain_rand.gravity_rand_interval * cfg.domain_rand.gravity_impulse_duration)
        cfg.domain_rand.lag_timesteps_rand_interval = np.ceil(cfg.domain_rand.lag_timesteps_rand_interval_s / self.dt)
    
    def get_segmentation_images(self, env_ids):
        segmentation_images = []
        for camera_name in self.cfg.perception.camera_names:
            segmentation_images = self.camera_sensors[camera_name].get_segmentation_images(env_ids)
        return segmentation_images

    def get_rgb_images(self, env_ids):
        rgb_images = {}
        for camera_name in self.cfg.perception.camera_names:
            rgb_images[camera_name] = self.camera_sensors[camera_name].get_rgb_images(env_ids)
        return rgb_images

    def get_depth_images(self, env_ids):
        depth_images = {}
        for camera_name in self.cfg.perception.camera_names:
            depth_images[camera_name] = self.camera_sensors[camera_name].get_depth_images(env_ids)
        return depth_images

    def initialize_cameras(self, env_ids):
        self.cams = {label: [] for label in self.cfg.perception.camera_names}
        self.camera_sensors = {}

        from globe_walking.go1_gym.sensors.attached_camera_sensor import AttachedCameraSensor

        for camera_label, camera_pose, camera_rpy in zip(self.cfg.perception.camera_names,
                                                             self.cfg.perception.camera_poses,
                                                             self.cfg.perception.camera_rpys):
            self.camera_sensors[camera_label] = AttachedCameraSensor(self)
            self.camera_sensors[camera_label].initialize(camera_label, camera_pose, camera_rpy, env_ids=env_ids)
        
    def set_lighting(self):
        light_index = 0
        intensity = gymapi.Vec3(0.5, 0.5, 0.5)
        ambient = gymapi.Vec3(0.2, 0.2, 0.2)
        direction = gymapi.Vec3(0.01, 0.01, 1.0)
        self.gym.set_light_parameters(self.sim, light_index, intensity, ambient, direction)

    def set_camera(self, position, lookat):
        """ Set camera position and direction
        """
        if position is not None and lookat is not None:
            cam_pos = gymapi.Vec3(position[0], position[1], position[2])
            cam_target = gymapi.Vec3(lookat[0], lookat[1], lookat[2])
            self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)
        else:
            bx, by, bz = self.root_states[self.robot_actor_idxs[0], 0], self.root_states[self.robot_actor_idxs[0], 1], self.root_states[self.robot_actor_idxs[0], 2]
            target_loc = [bx, by, bz]
            cam_distance = [2.0, -2.0, 2.0]

            cam_pos = gymapi.Vec3(target_loc[0] + cam_distance[0], target_loc[1] + cam_distance[1], target_loc[2] + cam_distance[2])
            cam_target = gymapi.Vec3(target_loc[0], target_loc[1], target_loc[2])
            self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)
