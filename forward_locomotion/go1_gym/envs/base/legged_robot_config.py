from params_proto import PrefixProto, ParamsProto
import os
import random


class Cfg(PrefixProto, cli=False):
    class env(PrefixProto, cli=False):
        num_envs = 4096
        num_observations = 235
        num_privileged_obs = 18
        privileged_future_horizon = 1
        num_actions = 12
        num_observation_history = 15
        env_spacing = 3.  # not used with heightfields/trimeshes 
        send_timeouts = True  # send time out information to the algorithm
        episode_length_s = 20  # episode length in seconds
        observe_vel = True
        observe_only_ang_vel = False
        observe_only_lin_vel = False
        observe_yaw = False
        observe_command = False
        record_video = True

        priv_observe_friction = True
        priv_observe_restitution = True
        priv_observe_base_mass = True
        priv_observe_com_displacement = True
        priv_observe_motor_strength = True
        priv_observe_Kp_factor = True
        priv_observe_Kd_factor = True

    class terrain(PrefixProto, cli=False):
        mesh_type = 'trimesh'  # "heightfield" # none, plane, heightfield or trimesh
        horizontal_scale = 0.1  # [m]
        vertical_scale = 0.005  # [m]
        border_size = 0  # 25 # [m]
        curriculum = True
        static_friction = 1.0
        dynamic_friction = 1.0
        restitution = 0.0
        terrain_noise_magnitude = 0.1
        # rough terrain only:
        terrain_smoothness = 0.005
        measure_heights = True
        # 1mx1.6m rectangle (without center line)
        measured_points_x = [-0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        measured_points_y = [-0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5]
        selected = False  # select a unique terrain type and pass all arguments
        terrain_kwargs = None  # Dict of arguments for selected terrain
        min_init_terrain_level = 0
        max_init_terrain_level = 5  # starting curriculum state
        terrain_length = 8.
        terrain_width = 8.
        num_rows = 10  # number of terrain rows (levels)
        num_cols = 20  # number of terrain cols (types)
        # terrain types: [smooth slope, rough slope, stairs up, stairs down, discrete]
        terrain_proportions = [0.1, 0.1, 0.35, 0.25, 0.2]
        # trimesh only:
        slope_treshold = 0.75  # slopes above this threshold will be corrected to vertical surfaces
        difficulty_scale = 1.
        x_init_range = 1.
        y_init_range = 1.
        x_init_offset = 0.
        y_init_offset = 0.
        teleport_robots = True
        teleport_thresh = 2.0
        max_platform_height = 0.2

    class commands_original(PrefixProto, cli=False):
        """Original command curriculum with x, y, and yaw velocities"""
        command_curriculum = False
        max_reverse_curriculum = 1.
        max_forward_curriculum = 1.
        forward_curriculum_threshold = 0.8
        yaw_command_curriculum = False
        max_yaw_curriculum = 1.
        yaw_curriculum_threshold = 0.5
        num_commands = 4
        resampling_time = 10.  # time before command are changed[s]
        heading_command = True  # if true: compute ang vel command from heading error
        global_reference = False

        num_lin_vel_bins = 20
        lin_vel_step = 0.3
        num_ang_vel_bins = 20
        ang_vel_step = 0.3
        distribution_update_extension_distance = 1
        curriculum_seed = 100

        # class ranges(ParamsProto, cli=False, prefix="commands.ranges"):
        lin_vel_x = [-1.0, 1.0]  # min max [m/s]
        lin_vel_y = [-1.0, 1.0]  # min max [m/s]
        ang_vel_yaw = [-1, 1]  # min max [rad/s]
        body_height_cmd = [-0.05, 0.05]
        impulse_height_commands = False

        limit_vel_x = [-10.0, 10.0]
        limit_vel_y = [-0.6, 0.6]
        limit_vel_yaw = [-10.0, 10.0]

        heading = [-3.14, 3.14]

    class commands_constrained(PrefixProto, cli=False):
        """Constrained curriculum that only allows forward velocity"""
        command_curriculum = False
        max_reverse_curriculum = 0.
        max_forward_curriculum = 1.
        forward_curriculum_threshold = 0.8
        yaw_command_curriculum = False
        max_yaw_curriculum = 0.
        yaw_curriculum_threshold = 0.5
        num_commands = 4
        resampling_time = 10.  # time before command are changed[s]
        heading_command = True  # if true: compute ang vel command from heading error
        global_reference = False

        num_lin_vel_bins = 20
        lin_vel_step = 0.3
        num_ang_vel_bins = 20
        ang_vel_step = 0.3
        distribution_update_extension_distance = 1
        curriculum_seed = 100

        # class ranges(ParamsProto, cli=False, prefix="commands.ranges"):
        lin_vel_x = [0.0, 1.0]  # min max [m/s]
        lin_vel_y = [0.0, 0.0]  # min max [m/s]
        ang_vel_yaw = [0, 0]  # min max [rad/s]
        body_height_cmd = [-0.05, 0.05]
        impulse_height_commands = False

        limit_vel_x = [0.0, 10.0]
        limit_vel_y = [0.0, 0.0]
        limit_vel_yaw = [0.0, 0.0]

        heading = [0.0, 0.0]

    class init_state(PrefixProto, cli=False):
        pos = [0.0, 0.0, 1.]  # x,y,z [m]
        rot = [0.0, 0.0, 0.0, 1.0]  # x,y,z,w [quat]
        lin_vel = [0.0, 0.0, 0.0]  # x,y,z [m/s]
        ang_vel = [0.0, 0.0, 0.0]  # x,y,z [rad/s]
        # target angles when action = 0.0
        default_joint_angles = {"joint_a": 0., "joint_b": 0.}

    class control(PrefixProto, cli=False):
        control_type = 'P'  # P: position, V: velocity, T: torques
        # PD Drive parameters:
        stiffness = {'joint_a': 10.0, 'joint_b': 15.}  # [N*m/rad]
        damping = {'joint_a': 1.0, 'joint_b': 1.5}  # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.5
        hip_scale_reduction = 1.0
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4

    class asset(PrefixProto, cli=False):
        file = ""
        foot_name = "None"  # name of the feet bodies, used to index body state and contact force tensors
        penalize_contacts_on = []
        terminate_after_contacts_on = []
        disable_gravity = False
        # merge bodies connected by fixed joints. Specific fixed joints can be kept by adding " <... dont_collapse="true">
        collapse_fixed_joints = True
        fix_base_link = False  # fixe the base of the robot
        default_dof_drive_mode = 3  # see GymDofDriveModeFlags (0 is none, 1 is pos tgt, 2 is vel tgt, 3 effort)
        self_collisions = 0  # 1 to disable, 0 to enable...bitwise filter
        # replace collision cylinders with capsules, leads to faster/more stable simulation
        replace_cylinder_with_capsule = True
        flip_visual_attachments = True  # Some .obj meshes must be flipped from y-up to z-up

        density = 0.001
        angular_damping = 0.
        linear_damping = 0.
        max_angular_velocity = 1000.
        max_linear_velocity = 1000.
        armature = 0.
        thickness = 0.01

    class domain_rand_original(PrefixProto, cli=False):
        rand_interval_s = 10
        randomize_base_mass = True
        added_mass_range = [-1, 3]
        push_robots = False
        push_interval_s = 15
        max_push_vel_xy = 1.
        max_push_vel_xy = 0.5
        randomize_friction = True
        friction_range = [0.05, 4.5]
        randomize_restitution = True
        restitution_range = [0.0, 1.0]
        restitution = 0.5
        randomize_com_displacement = True
        com_displacement_range = [-0.1, 0.1]
        randomize_motor_strength = True
        motor_strength_range = [0.9, 1.1]
        randomize_Kp_factor = False
        Kp_factor_range = None
        randomize_Kd_factor = False
        Kd_factor_range = None

        # Extra stuff added by @wliang
        randomize_gravity = False
        gravity_rand_interval_s = 10
        randomize_rolling_friction = False
        randomize_torsion_friction = False
        randomize_dof_stiffness = False
        randomize_dof_damping = False
        randomize_dof_friction = False
        randomize_dof_armature = False

    class domain_rand_eureka(PrefixProto, cli=False):
        rand_interval_s = 10
        friction_range = None
        rolling_friction_range = None
        torsion_friction_range = None
        restitution_range = None
        added_mass_range = None
        com_displacement_range = None
        motor_strength_range = None
        Kp_factor_range = None
        Kd_factor_range = None
        dof_stiffness_range = None
        dof_damping_range = None
        dof_friction_range = None
        dof_armature_range = None
        push_vel_xy_range = None  # Hack to format this as range
        max_push_vel_xy = None
        push_interval_s = 15
        gravity_range = None
        gravity_rand_interval_s = 10

# INSERT EUREKA DR HERE

        randomize_friction = False if friction_range is None else True
        randomize_rolling_friction = False if rolling_friction_range is None else True
        randomize_torsion_friction = False if torsion_friction_range is None else True
        randomize_restitution = False if restitution_range is None else True
        randomize_base_mass = False if added_mass_range is None else True
        randomize_com_displacement = False if com_displacement_range is None else True
        randomize_motor_strength = False if motor_strength_range is None else True
        randomize_Kp_factor = False if Kp_factor_range is None else True
        randomize_Kd_factor = False if Kd_factor_range is None else True
        randomize_dof_stiffness = False if dof_stiffness_range is None else True
        randomize_dof_damping = False if dof_damping_range is None else True
        randomize_dof_friction = False if dof_friction_range is None else True
        randomize_dof_armature = False if dof_armature_range is None else True
        max_push_vel_xy = None if push_vel_xy_range is None else push_vel_xy_range[1]
        push_robots = False if max_push_vel_xy is None else True
        randomize_gravity = False if gravity_range is None else True

    class domain_rand_off(PrefixProto, cli=False):
        rand_interval_s = 10
        friction_range = None
        rolling_friction_range = None
        torsion_friction_range = None
        restitution_range = None
        added_mass_range = None
        com_displacement_range = None
        motor_strength_range = None
        Kp_factor_range = None
        Kd_factor_range = None
        dof_stiffness_range = None
        dof_damping_range = None
        dof_friction_range = None
        dof_armature_range = None
        push_vel_xy_range = None  # Hack to format this as range
        max_push_vel_xy = None
        push_interval_s = 15
        gravity_range = None
        gravity_rand_interval_s = 10

        randomize_friction = False
        randomize_rolling_friction = False
        randomize_torsion_friction = False
        randomize_restitution = False
        randomize_base_mass = False
        randomize_com_displacement = False
        randomize_motor_strength = False
        randomize_Kp_factor = False
        randomize_Kd_factor = False
        randomize_dof_stiffness = False 
        randomize_dof_damping = False
        randomize_dof_friction = False
        randomize_dof_armature = False
        push_robots = False
        randomize_gravity = False
    
    class rewards_original(PrefixProto, cli=False):
        """Original rewards from rapid loco"""
        only_positive_rewards = True  # if true negative total rewards are clipped at zero (avoids early termination problems)
        tracking_sigma = 0.25  # tracking reward = exp(-error^2/sigma)
        tracking_sigma_lat = 0.25  # tracking reward = exp(-error^2/sigma)
        tracking_sigma_long = 0.25  # tracking reward = exp(-error^2/sigma)
        tracking_sigma_yaw = 0.25  # tracking reward = exp(-error^2/sigma)
        soft_dof_pos_limit = 1.  # percentage of urdf limits, values above this limit are penalized
        soft_dof_vel_limit = 1.
        soft_torque_limit = 1.
        base_height_target = 1.
        max_contact_force = 100.  # forces above this value are penalized
        use_terminal_body_height = False
        terminal_body_height = 0.20
        reward_container_name = "OriginalReward"

        class scales(ParamsProto, cli=False, prefix="rewards.scales"):
            termination = -0.0
            # forward_lin_vel = 2.0
            # forward_ang_vel = 0.5
            tracking_lin_vel = 1.0
            tracking_ang_vel = 0.5
            lin_vel_z = -2.0
            ang_vel_xy = -0.05
            orientation = -0.
            torques = -0.00001
            dof_vel = -0.
            dof_acc = -2.5e-7
            base_height = -0.
            feet_air_time = 1.0
            collision = -1.
            feet_stumble = -0.0
            action_rate = -0.01
            stand_still = -0.
            tracking_lin_vel_lat = 0.
            tracking_lin_vel_long = 0.

    class rewards_eureka(PrefixProto, cli=False):
        tracking_sigma = 0.25          # Can be used for command curriculum, if enabled
        tracking_sigma_yaw = 0.25      # Can be used for command curriculum, if enabled
        use_terminal_body_height = False
        terminal_body_height = 0.20
        reward_container_name = "EurekaReward"
        target_velocity = 2.0
        class scales(ParamsProto, cli=False, prefix="rewards.scales"):
            tracking_lin_vel = 1.0     # Used for command curriculum, if enabled
            tracking_ang_vel = 0.5     # Used for command curriculum, if enabled
            pass

    class normalization(PrefixProto, cli=False):
        class obs_scales(PrefixProto, cli=False):
            lin_vel = 2.0
            ang_vel = 0.25
            dof_pos = 1.0
            dof_vel = 0.05
            height_measurements = 5.0
            body_height_cmd = 2.0

        clip_observations = 100.
        clip_actions = 100.

    class noise(PrefixProto, cli=False):
        add_noise = True
        noise_level = 1.0  # scales other values

        class noise_scales(PrefixProto, cli=False):
            dof_pos = 0.01
            dof_vel = 1.5
            lin_vel = 0.1
            ang_vel = 0.2
            gravity = 0.05
            height_measurements = 0.1

    # viewer camera:
    class viewer(PrefixProto, cli=False):
        ref_env = 0
        pos = [10, 0, 6]  # [m]
        lookat = [11., 5, 3.]  # [m]

    class sim(PrefixProto, cli=False):
        dt = 0.005
        substeps = 1
        gravity = [0., 0., -9.81]  # [m/s^2]
        up_axis = 1  # 0 is y, 1 is z

        use_gpu_pipeline = True

        class physx(PrefixProto, cli=False):
            num_threads = 10
            solver_type = 1  # 0: pgs, 1: tgs
            num_position_iterations = 4
            num_velocity_iterations = 0
            contact_offset = 0.01  # [m]
            rest_offset = 0.0  # [m]
            bounce_threshold_velocity = 0.5  # 0.5 [m/s]
            max_depenetration_velocity = 1.0
            max_gpu_contact_pairs = 2 ** 23  # 2**24 -> needed for 8000 envs and more
            default_buffer_size_multiplier = 5
            contact_collection = 2  # 0: never, 1: last sub-step, 2: all sub-steps (default=2)


def set_seed(seed, torch_deterministic=False, rank=0):
    # From isaacgymenvs.utils.utils

    import os
    import random
    import numpy as np
    import torch

    """ set seed across modules """
    if seed == -1 and torch_deterministic:
        seed = 42 + rank
    elif seed == -1:
        seed = np.random.randint(0, 10000)
    else:
        seed = seed + rank

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if torch_deterministic:
        # refer to https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True)
    else:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    return seed

# This function sets the seed whenever Cfg is imported
# To overwrite, call set_seed() afterwards
set_seed(0, torch_deterministic=False)