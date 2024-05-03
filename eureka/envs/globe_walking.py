class LeggedRobot(BaseTask):
    """Rest of environment ommitted"""

    def _init_buffers(self):
        # Buffer has shape (num_actors, 13). State for each actor root contains position([0:3]), rotation([3:7]), linear velocity([7:10]), and angular velocity([10:13]).
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        # Buffer has shape (num_dofs, 2). Each DOF state contains position and velocity.
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        # Buffer has shape (num_rigid_bodies, 3). Each contact force state contains one value for each X, Y, Z axis.
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)
        # Buffer has shape (num_rigid_bodies, 13). State for each rigid body contains position([0:3]), rotation([3:7]), linear velocity([7:10]), and angular velocity([10:13]).
        rigid_body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.render_all_camera_sensors(self.sim)

        # This is in the global frame
        self.root_states = gymtorch.wrap_tensor(actor_root_state)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.net_contact_forces = gymtorch.wrap_tensor(net_contact_forces)[:self.num_envs * self.num_bodies, :]
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]
        self.base_pos = self.root_states[self.robot_actor_idxs, 0:3]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]
        self.base_quat = self.root_states[self.robot_actor_idxs, 3:7]
        self.rigid_body_state = gymtorch.wrap_tensor(rigid_body_state)[:self.num_envs*self.total_rigid_body_num,:].view(self.num_envs,self.total_rigid_body_num,13)[:,0:self.num_bodies, :]#.contiguous().view(self.num_envs*self.num_bodies,13)
        self.rigid_body_state_object = gymtorch.wrap_tensor(rigid_body_state)[:self.num_envs*self.total_rigid_body_num,:].view(self.num_envs,self.total_rigid_body_num,13)[:,self.num_bodies:self.num_bodies + self.num_object_bodies, :]#.contiguous().view(self.num_envs*self.num_bodies,13)
        self.foot_velocities = self.rigid_body_state.view(self.num_envs, -1, 13)[:, self.feet_indices, 7:10]
        self.foot_positions = self.rigid_body_state.view(self.num_envs, -1, 13)[:, self.feet_indices, 0:3]
        self.prev_base_pos = self.base_pos.clone()
        self.prev_foot_velocities = self.foot_velocities.clone()

        self.lag_buffer = [torch.zeros_like(self.dof_pos) for i in range(self.cfg.domain_rand.lag_timesteps+1)]

        self.contact_forces = gymtorch.wrap_tensor(net_contact_forces)[:self.num_envs*self.total_rigid_body_num,:].view(self.num_envs,self.total_rigid_body_num,3)[:,0:self.num_bodies, :]

        self.object_pos_world_frame = self.root_states[self.object_actor_idxs, 0:3]
        robot_object_vec = self.asset.get_local_pos()
        self.object_local_pos = quat_rotate_inverse(self.base_quat, robot_object_vec)
        self.object_local_pos[:, 2] = 0.0*torch.ones(self.num_envs, dtype=torch.float,
                                    device=self.device, requires_grad=False)

        self.last_object_local_pos = torch.clone(self.object_local_pos)
        self.object_lin_vel = self.asset.get_lin_vel()
        self.object_ang_vel = self.asset.get_ang_vel()
         
        self.common_step_counter = 0
        self.extras = {}

        self.measured_heights = 0

        self.gravity_vec = to_torch(get_axis_params(-1., self.up_axis_idx), device=self.device).repeat( (self.num_envs, 1))
        self.forward_vec = to_torch([1., 0., 0.], device=self.device).repeat((self.num_envs, 1))
        self.torques = torch.zeros(self.num_envs, self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        self.forces = torch.zeros(self.num_envs, self.total_rigid_body_num, 3, dtype=torch.float, device=self.device, requires_grad=False)
        self.p_gains = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        self.d_gains = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        self.actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_last_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.joint_pos_target = torch.zeros(self.num_envs, self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_joint_pos_target = torch.zeros(self.num_envs, self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_last_joint_pos_target = torch.zeros(self.num_envs, self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_dof_vel = torch.zeros_like(self.dof_vel)
        self.last_root_vel = torch.zeros_like(self.root_states[self.robot_actor_idxs, 7:13])
        self.path_distance = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self.past_base_pos = self.base_pos.clone()  # Updated only when environment is reset, use self.prev_base_pos for delta position within a rollout

        self.feet_air_time = torch.zeros(self.num_envs, self.feet_indices.shape[0], dtype=torch.float, device=self.device, requires_grad=False)
        self.last_contacts = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.bool, device=self.device, requires_grad=False)
        self.last_contact_filt = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.bool, device=self.device, requires_grad=False)
        # These are in the local frame of the robot
        self.base_lin_vel = quat_rotate_inverse(self.base_quat, self.root_states[self.robot_actor_idxs, 7:10])
        self.base_ang_vel = quat_rotate_inverse(self.base_quat, self.root_states[self.robot_actor_idxs, 10:13])
        
        self.projected_gravity = quat_rotate_inverse(self.base_quat, self.gravity_vec)

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

        actuator_path = f'{os.path.dirname(os.path.dirname(os.path.realpath(__file__)))}/../../resources/actuator_nets/unitree_go1.pt'
        actuator_network = torch.jit.load(actuator_path, map_location=self.device)

        def eval_actuator_network(joint_pos, joint_pos_last, joint_pos_last_last, joint_vel, joint_vel_last, joint_vel_last_last):
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
        # Rest of function ommitted
        self.ball_radius = self.default_ball_radius * torch.ones(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)