import torch
    
class OriginalReward():
    def __init__(self, env):
        self.env = env

    def load_env(self, env):
        self.env = env
    # ------------ reward functions----------------
    def _reward_lin_vel_z(self):
        # Penalize z axis base linear velocity
        return torch.square(self.env.base_lin_vel[:, 2])

    def _reward_ang_vel_xy(self):
        # Penalize xy axes base angular velocity
        return torch.sum(torch.square(self.env.base_ang_vel[:, :2]), dim=1)

    def _reward_orientation(self):
        # Penalize non flat base orientation
        return torch.sum(torch.square(self.env.projected_gravity[:, :2]), dim=1)

    def _reward_base_height(self):
        # Penalize base height away from target
        base_height = torch.mean(self.env.root_states[:, 2].unsqueeze(1) - self.env.measured_heights, dim=1)
        return torch.square(base_height - self.env.cfg.rewards.base_height_target)

    def _reward_torques(self):
        # Penalize torques
        return torch.sum(torch.square(self.env.torques), dim=1)

    def _reward_energy(self):
        # Penalize torques
        return torch.sum(torch.multiply(self.env.torques, self.env.dof_vel), dim=1)

    def _reward_energy_expenditure(self):
        # Penalize torques
        return torch.sum(torch.clip(torch.multiply(self.env.torques, self.env.dof_vel), 0, 1e30), dim=1)

    def _reward_dof_vel(self):
        # Penalize dof velocities
        return torch.sum(torch.square(self.env.dof_vel), dim=1)

    def _reward_dof_acc(self):
        # Penalize dof accelerations
        return torch.sum(torch.square((self.env.last_dof_vel - self.env.dof_vel) / self.env.dt), dim=1)

    def _reward_action_rate(self):
        # Penalize changes in actions
        return torch.sum(torch.square(self.env.last_actions - self.env.actions), dim=1)

    def _reward_collision(self):
        # Penalize collisions on selected bodies
        return torch.sum(1. * (torch.norm(self.env.contact_forces[:, self.env.penalised_contact_indices, :], dim=-1) > 0.1),
                         dim=1)

    def _reward_termination(self):
        # Terminal reward / penalty
        return self.env.reset_buf * ~self.env.time_out_buf

    def _reward_survival(self):
        # Survival reward / penalty
        return ~(self.env.reset_buf * ~self.env.time_out_buf)

    def _reward_dof_pos_limits(self):
        # Penalize dof positions too close to the limit
        out_of_limits = -(self.env.dof_pos - self.env.dof_pos_limits[:, 0]).clip(max=0.)  # lower limit
        out_of_limits += (self.env.dof_pos - self.env.dof_pos_limits[:, 1]).clip(min=0.)
        return torch.sum(out_of_limits, dim=1)

    def _reward_dof_vel_limits(self):
        # Penalize dof velocities too close to the limit
        # clip to max error = 1 rad/s per joint to avoid huge penalties
        return torch.sum(
            (torch.abs(self.env.dof_vel) - self.env.dof_vel_limits * self.env.cfg.rewards.soft_dof_vel_limit).clip(min=0., max=1.),
            dim=1)

    def _reward_torque_limits(self):
        # penalize torques too close to the limit
        return torch.sum(
            (torch.abs(self.env.torques) - self.env.torque_limits * self.env.cfg.rewards.soft_torque_limit).clip(min=0.), dim=1)
    
    def _reward_tracking_lin_vel(self):
        # Tracking of linear velocity commands (xy axes)
        if self.env.cfg.commands.global_reference:
            lin_vel_error = torch.sum(torch.square(self.env.commands[:, :2] - self.env.root_states[:, 7:9]), dim=1)
        else:
            lin_vel_error = torch.sum(torch.square(self.env.commands[:, :2] - self.env.base_lin_vel[:, :2]), dim=1)
        return torch.exp(-lin_vel_error / self.env.cfg.rewards.tracking_sigma)

    # def _reward_tracking_lin_vel_long(self):
    #     # Tracking of linear velocity commands (xy axes)
    #     lin_vel_error = torch.sum(torch.square(self.env.commands[:, 0] - self.env.base_lin_vel[:, 0]), dim=1)
    #     return torch.exp(-lin_vel_error / self.env.cfg.rewards.tracking_sigma_long)
    #
    # def _reward_tracking_lin_vel_lat(self):
    #     # Tracking of linear velocity commands (xy axes)
    #     lin_vel_error = torch.sum(torch.square(self.env.commands[:, 1] - self.env.base_lin_vel[:, 1]), dim=1)
    #     return torch.exp(-lin_vel_error / self.env.cfg.rewards.tracking_sigma_lat)

    # def _reward_clipped_forward_progress(self):
    #     # Tracking of linear velocity commands (xy axes)
    #     forward_progress = self.env.base_lin_vel[:, 0] * self.env.dt
    #     clipped_forward_progress = forward_progress.clip(max=self.env.cfg.rewards.max_velocity * self.env.dt)
    #     return clipped_forward_progress
    #
    # def _reward_clipped_global_forward_progress(self):
    #     # Tracking of linear velocity commands (xy axes)
    #     forward_progress = self.env.root_states[:, 7] * self.env.dt
    #     clipped_forward_progress = forward_progress.clip(max=self.env.cfg.rewards.max_velocity * self.env.dt)
    #     return clipped_forward_progress

    # def _reward_jump(self):
    #     body_height = torch.mean(self.env.root_states[:, 2:3] - self.env.measured_heights, dim=-1)
    #     jump_height_target = self.env.commands[:, 3] + self.env.cfg.rewards.base_height_target
    #     reward = - torch.square(body_height - jump_height_target)
    #     return reward

    def _reward_tracking_ang_vel(self):
        # Tracking of angular velocity commands (yaw) 
        ang_vel_error = torch.square(self.env.commands[:, 2] - self.env.base_ang_vel[:, 2])
        return torch.exp(-ang_vel_error / self.env.cfg.rewards.tracking_sigma_yaw)

    def _reward_feet_air_time(self):
        # Reward long steps
        # Need to filter the contacts because the contact reporting of PhysX is unreliable on meshes
        contact = self.env.contact_forces[:, self.env.feet_indices, 2] > 1.
        contact_filt = torch.logical_or(contact, self.env.last_contacts)
        self.env.last_contacts = contact
        first_contact = (self.env.feet_air_time > 0.) * contact_filt
        self.env.feet_air_time += self.env.dt
        rew_airTime = torch.sum((self.env.feet_air_time - 0.5) * first_contact,
                                dim=1)  # reward only on first contact with the ground
        rew_airTime *= torch.norm(self.env.commands[:, :2], dim=1) > 0.1  # no reward for zero command
        # rew_airTime *= torch.norm(self.env.base_lin_vel[:, :2], dim=1) > 0.1  # no reward for zero movement
        self.env.feet_air_time *= ~contact_filt
        return rew_airTime

    def _reward_stumble(self):
        # Penalize feet hitting vertical surfaces
        return torch.any(torch.norm(self.env.contact_forces[:, self.env.feet_indices, :2], dim=2) > \
                         5 * torch.abs(self.env.contact_forces[:, self.env.feet_indices, 2]), dim=1)

    def _reward_stand_still(self):
        # Penalize motion at zero commands
        return torch.sum(torch.abs(self.env.dof_pos - self.env.default_dof_pos), dim=1) * (
                torch.norm(self.env.commands[:, :2], dim=1) < 0.1)

    def _reward_feet_contact_forces(self):
        # penalize high contact forces
        return torch.sum((torch.norm(self.env.contact_forces[:, self.env.feet_indices, :],
                                     dim=-1) - self.env.cfg.rewards.max_contact_force).clip(min=0.), dim=1)

    # def _reward_contact_prediction(self):
    #     contact = (self.env.contact_forces[:, self.env.feet_indices, 2] > 1.) * 1
    #     contact_prediction = torch.sigmoid(self.env.actions[:, 12:])
    #     # return torch.exp(-torch.norm(contact - contact_prediction, dim=-1))
    #     anneal_steps = 200 * 25
    #     anneal_multiplier = 10 ** (-max(0, anneal_steps - self.env.common_step_counter) / anneal_steps)
    #     # print(anneal_multiplier)
    #     return torch.sum(contact * torch.log(contact_prediction) + (1 - contact) * torch.log(1 - contact_prediction),
    #                      dim=1) * anneal_multiplier
    #
    # def _reward_stance_state_prediction(self):
    #     stance_state = (torch.norm(self.env.foot_velocities, dim=2) < 0.1) * 1.
    #     stance_state_prediction = torch.sigmoid(self.env.actions[:, 12:])
    #     # return torch.exp(-torch.norm(contact - contact_prediction, dim=-1))
    #     anneal_steps = 200 * 25
    #     anneal_multiplier = 10 ** (-max(0, anneal_steps - self.env.common_step_counter) / anneal_steps)
    #     # print(anneal_multiplier)
    #     return torch.sum(stance_state * torch.log(stance_state_prediction) + (1 - stance_state) * torch.log(
    #         1 - stance_state_prediction), dim=1) * anneal_multiplier
