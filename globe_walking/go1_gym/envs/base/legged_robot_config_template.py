from params_proto import PrefixProto, ParamsProto
import os 
import yaml


class Cfg(PrefixProto, cli=False):
    multi_gpu = False
    class env_mini(PrefixProto, cli=False):
        num_envs = 1000
        num_observations = 235
        # if not None a privilige_obs_buf will be returned by step() (critic obs for assymetric training). None is returned otherwise
        num_privileged_obs = 6
        num_actions = 12
        num_observation_history = 15
        env_spacing = 3.  # not used with heightfields/trimeshes
        send_timeouts = True  # send time out information to the algorithm
        episode_length_s = 20  # episode length in seconds
        record_video = True
        recording_width_px = 360
        recording_height_px = 240
        recording_mode = "COLOR"
        num_recording_envs = 1
        add_balls = True

    class env_full(PrefixProto, cli=False):
        num_envs = 4096
        num_observations = 235
        # if not None a privilige_obs_buf will be returned by step() (critic obs for assymetric training). None is returned otherwise
        num_privileged_obs = 11
        num_actions = 12
        num_observation_history = 15
        env_spacing = 3.  # not used with heightfields/trimeshes
        send_timeouts = True  # send time out information to the algorithm
        episode_length_s = 20  # episode length in seconds
        record_video = True
        recording_width_px = 360
        recording_height_px = 240
        recording_mode = "COLOR"
        num_recording_envs = 1
        add_balls = True
    
    class robot(PrefixProto, cli=False):
        name = "go1"

    class ball(PrefixProto, cli=False):
        asset = "ball"
        ball_init_pos = [0.0, 0.0, 0.0]
        ball_init_rot = [0, 0, 0, 1]
        ball_init_lin_vel = [0, 0, 0]
        ball_init_ang_vel = [0, 0, 0]
        init_pos_range = [0.0, 0.0, 0.0]
        init_vel_range = [0.0, 0.0, 0.0]
        pos_reset_prob = 0.0002
        vel_reset_prob = 0.0008
        pos_reset_range = [1.0, 1.0, 0.0]
        vel_reset_range = [0.3, 0.3, 0.3]
        vision_receive_prob = 0.7
        
        limit_movement = False

    class sensors_mini(PrefixProto, cli=False):
        sensor_names = [
            "OrientationSensor",
            "JointPositionSensor",
            "JointVelocitySensor",
            "ActionSensor",
            "LastActionSensor",
            "ClockSensor",
            "YawSensor",
        ]
        sensor_args = {
            "OrientationSensor": {},
            "JointPositionSensor": {},
            "JointVelocitySensor": {},
            "ActionSensor": {},
            "LastActionSensor": {"delay": 1},
            "ClockSensor": {},
            "YawSensor": {},
        }
        privileged_sensor_names = {
            "BodyVelocitySensor": {},
            "ObjectVelocitySensor": {},
        }
        privileged_sensor_args = {
            "BodyVelocitySensor": {},
            "ObjectVelocitySensor": {},
        }

    class sensors_full(PrefixProto, cli=False):
        sensor_names = [
            "OrientationSensor",
            "JointPositionSensor",
            "JointVelocitySensor",
            "ActionSensor",
            "LastActionSensor",
            "ClockSensor",
            "YawSensor",
        ]
        sensor_args = {
            "OrientationSensor": {},
            "JointPositionSensor": {},
            "JointVelocitySensor": {},
            "ActionSensor": {},
            "LastActionSensor": {"delay": 1},
            "ClockSensor": {},
            "YawSensor": {},
        }
        privileged_sensor_names = {
            "ObjectSensor": {},
            "BodyVelocitySensor": {},
            "ObjectVelocitySensor": {},
            "RestitutionSensor": {},
            "FrictionSensor": {},
        }
        privileged_sensor_args = {
            "ObjectSensor": {},
            "BodyVelocitySensor": {},
            "ObjectVelocitySensor": {},
            "RestitutionSensor": {},
            "FrictionSensor": {},
        }
        

    class terrain_mini(PrefixProto, cli=False):
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

        min_init_terrain_level = 0
        max_init_terrain_level = 5  # starting curriculum state
        terrain_length = 8.
        terrain_width = 8.
        num_rows = 10  # number of terrain rows (levels)
        num_cols = 20  # number of terrain cols (types)
        num_border_boxes = 0
        # terrain types: [smooth slope, rough slope, stairs up, stairs down, discrete]
        terrain_proportions = [0.1, 0.1, 0.35, 0.25, 0.2]
        # trimesh only:
        slope_treshold = 0.75  # slopes above this threshold will be corrected to vertical surfaces
        difficulty_scale = 1.
        x_init_range = 0.05
        y_init_range = 0.05
        yaw_init_range = 3.14
        x_init_offset = 0.
        y_init_offset = 0.
        teleport_robots = True
        teleport_thresh = 2.0
        max_platform_height = 0.2
        max_step_height = 0.26
        min_step_run = 0.25
        max_step_run = 0.4
        center_robots = False
        center_span = 5

    class terrain_full(PrefixProto, cli=False):
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

        min_init_terrain_level = 0
        max_init_terrain_level = 5  # starting curriculum state
        terrain_length = 8.
        terrain_width = 8.
        num_rows = 10  # number of terrain rows (levels)
        num_cols = 20  # number of terrain cols (types)
        num_border_boxes = 0
        # terrain types: [smooth slope, rough slope, stairs up, stairs down, discrete]
        terrain_proportions = [0.1, 0.1, 0.35, 0.25, 0.2]
        # trimesh only:
        slope_treshold = 0.75  # slopes above this threshold will be corrected to vertical surfaces
        difficulty_scale = 1.
        x_init_range = 0.1
        y_init_range = 0.1
        yaw_init_range = 3.14
        x_init_offset = 0.
        y_init_offset = 0.
        teleport_robots = True
        teleport_thresh = 2.0
        max_platform_height = 0.2
        max_step_height = 0.26
        min_step_run = 0.25
        max_step_run = 0.4
        center_robots = False
        center_span = 5

    class commands(PrefixProto, cli=False):
        command_curriculum = False
        max_reverse_curriculum = 1.
        max_forward_curriculum = 1.
        yaw_command_curriculum = False
        max_yaw_curriculum = 1.
        exclusive_command_sampling = False
        num_commands = 0
        resampling_time = 10.  # time before command are changed[s]
        subsample_gait = False
        gait_interval_s = 10.  # time between resampling gait params
        vel_interval_s = 10.
        jump_interval_s = 20.  # time between jumps
        jump_duration_s = 0.1  # duration of jump
        jump_height = 0.3
        heading_command = True  # if true: compute ang vel command from heading error
        global_reference = False
        observe_accel = False
        distributional_commands = False
        curriculum_type = "RewardThresholdCurriculum"
        lipschitz_threshold = 0.9

        num_lin_vel_bins = 20
        lin_vel_step = 0.3
        num_ang_vel_bins = 20
        ang_vel_step = 0.3
        distribution_update_extension_distance = 1
        curriculum_seed = 100

        lin_vel_x = [-1.0, 1.0]  # min max [m/s]
        lin_vel_y = [-1.0, 1.0]  # min max [m/s]
        ang_vel_yaw = [-1, 1]  # min max [rad/s]
        body_height_cmd = [-0.05, 0.05]
        impulse_height_commands = False

        limit_vel_x = [-10.0, 10.0]
        limit_vel_y = [-0.6, 0.6]
        limit_vel_yaw = [-10.0, 10.0]
        limit_body_height = [-0.05, 0.05]
        limit_gait_phase = [0, 0.01]
        limit_gait_offset = [0, 0.01]
        limit_gait_bound = [0, 0.01]
        limit_gait_frequency = [2.0, 2.01]
        limit_gait_duration = [0.49, 0.5]
        limit_footswing_height = [0.06, 0.061]
        limit_body_pitch = [0.0, 0.01]
        limit_body_roll = [0.0, 0.01]
        limit_aux_reward_coef = [0.0, 0.01]
        limit_compliance = [0.0, 0.01]
        limit_stance_width = [0.0, 0.01]
        limit_stance_length = [0.0, 0.01]

        num_bins_vel_x = 25
        num_bins_vel_y = 3
        num_bins_vel_yaw = 25
        num_bins_body_height = 1
        num_bins_gait_frequency = 11
        num_bins_gait_phase = 11
        num_bins_gait_offset = 2
        num_bins_gait_bound = 2
        num_bins_gait_duration = 3
        num_bins_footswing_height = 1
        num_bins_body_pitch = 1
        num_bins_body_roll = 1
        num_bins_aux_reward_coef = 1
        num_bins_compliance = 1
        num_bins_compliance = 1
        num_bins_stance_width = 1
        num_bins_stance_length = 1

        heading = [-3.14, 3.14]

        gait_phase_cmd_range = [0.0, 0.01]
        gait_offset_cmd_range = [0.0, 0.01]
        gait_bound_cmd_range = [0.0, 0.01]
        gait_frequency_cmd_range = [2.0, 2.01]
        gait_duration_cmd_range = [0.49, 0.5]
        footswing_height_range = [0.06, 0.061]
        body_pitch_range = [0.0, 0.01]
        body_roll_range = [0.0, 0.01]
        aux_reward_coef_range = [0.0, 0.01]
        compliance_range = [0.0, 0.01]
        stance_width_range = [0.0, 0.01]
        stance_length_range = [0.0, 0.01]

        exclusive_phase_offset = True
        binary_phases = False
        pacing_offset = False
        balance_gait_distribution = True
        gaitwise_curricula = True

    class curriculum_thresholds(PrefixProto, cli=False):
        tracking_lin_vel = 0.8  # closer to 1 is tighter
        tracking_ang_vel = 0.5
        tracking_contacts_shaped_force = 0.8  # closer to 1 is tighter
        tracking_contacts_shaped_vel = 0.8

    class init_state(PrefixProto, cli=False):
        pos = [0.0, 0.0, 0.3]  # x,y,z [m]
        rot = [0.0, 0.0, 0.0, 1.0]  # x,y,z,w [quat]
        lin_vel = [0.0, 0.0, 0.0]  # x,y,z [m/s]
        ang_vel = [0.0, 0.0, 0.0]  # x,y,z [rad/s]
        # target angles when action = 0.0
        default_joint_angles = {"joint_a": 0., "joint_b": 0.}

    class control(PrefixProto, cli=False):
        control_type = 'actuator_net' #'P'  # P: position, V: velocity, T: torques
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
    
    class domain_rand_eureka(PrefixProto, cli=False):
        # Setting default values
        randomize = True
        robot_friction_range = [1.0, 1.0]
        robot_restitution_range = [0.0, 0.0]
        robot_payload_mass_range = [0.0, 0.0]
        robot_com_displacement_range = [0.0, 0.0]
        robot_motor_strength_range = [1.0, 1.0]
        robot_motor_offset_range = [0.0, 0.0]
        robot_Kp_factor_range = [1.0, 1.0]
        robot_Kd_factor_range = [1.0, 1.0]

        ball_radius_range = [0.35, 0.45]  # Measured in real
        ball_mass_range = [1.0, 1.0]
        ball_friction_range = [1.0, 1.0]
        ball_restitution_range = [0.0, 0.0]
        ball_compliance_range = [0.0, 0.0]
        ball_inertia_multiplier_range = [5/3, 5/3]  # For hollow ball
        ball_spring_coefficient_range = [0.0, 0.0]
        ball_drag_range = [0.0, 0.0]
        ball_drag_rand_interval_s = 15

        terrain_ground_friction_range = [1.0, 1.0]
        terrain_ground_restitution_range = [0.0, 0.0]
        terrain_tile_roughness_range = [0.0, 0.0]

        push_robot_interval_s = 15
        robot_push_vel_range = [0.0, 0.0]
        push_ball_interval_s = 10
        ball_push_vel_range = [0.0, 0.0]

        gravity_range = [0.0, 0.0]
        gravity_rand_interval_s = 10

        lag_timesteps_range = [6, 6]
        lag_timesteps_rand_interval_s = 10

# INSERT EUREKA DR HERE
    
    class domain_rand_off(PrefixProto, cli=False):
        randomize = True
        robot_friction_range = [1.0, 1.0]
        robot_restitution_range = [0.0, 0.0]
        robot_payload_mass_range = [0.0, 0.0]
        robot_com_displacement_range = [0.0, 0.0]
        robot_motor_strength_range = [1.0, 1.0]
        robot_motor_offset_range = [0.0, 0.0]
        robot_Kp_factor_range = [1.0, 1.0]
        robot_Kd_factor_range = [1.0, 1.0]

        ball_radius_range = [0.45, 0.45]
        ball_mass_range = [1.0, 1.0]
        ball_friction_range = [1.0, 1.0]
        ball_restitution_range = [0.0, 0.0]
        ball_compliance_range = [0.0, 0.0]
        ball_inertia_multiplier_range = [5/3, 5/3]  # For hollow ball
        ball_spring_coefficient_range = [0.0, 0.0]
        ball_drag_range = [0.0, 0.0]
        ball_drag_rand_interval_s = 15

        terrain_ground_friction_range = [1.0, 1.0]
        terrain_ground_restitution_range = [0.0, 0.0]
        terrain_tile_roughness_range = [0.0, 0.0]

        push_robot_interval_s = 15
        robot_push_vel_range = [0.0, 0.0]
        push_ball_interval_s = 10
        ball_push_vel_range = [0.0, 0.0]

        gravity_range = [0.0, 0.0]
        gravity_rand_interval_s = 10

        lag_timesteps_range = [6, 6]
        lag_timesteps_rand_interval_s = 10

    class rewards(PrefixProto, cli=False):
        reward_container_name = "EurekaReward"
        soft_dof_pos_limit = 1.  # Used to set DOF pos limits
        use_terminal_body_height = False
        terminal_body_height = 0.20
        use_terminal_roll_pitch = False
        terminal_body_ori = 0.5
        
        constrict = False
        constrict_indices = []
        constrict_ranges = [[]]
        constrict_after = 0

    class reward_scales(ParamsProto, cli=False):
        termination = -0.0
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
        tracking_contacts = 0.
        tracking_contacts_shaped = 0.
        tracking_contacts_shaped_force = 0.
        tracking_contacts_shaped_vel = 0.
        jump = 0.0
        energy = 0.0
        energy_expenditure = 0.0
        survival = 0.0
        dof_pos_limits = 0.0
        dof_vel_limits = 0.0
        torque_limits = 0.0
        feet_contact_forces = 0.
        feet_slip = 0.
        feet_accel = 0.
        dof_pos = 0.
        action_smoothness_1 = 0.
        action_smoothness_2 = 0.
        base_motion = 0.
        feet_impact_vel = 0.0
        raibert_heuristic = 0.0
        dribbling_robot_ball_vel = 0.0
        dribbling_robot_ball_pos = 0.0
        dribbling_ball_vel = 0.0
        dribbling_robot_ball_yaw = 0.0
        dribbling_ball_vel_norm = 0.0
        dribbling_ball_vel_angle = 0.0
        gripper_handle_pos = 0.0
        gripper_handle_height = 0.0
        turn_handle = 0.0
        open_door = 0.0
        robot_door_pos = 0.0
        robot_door_ori = 0.0
        estimation_bonus = 0.0

    class normalization(PrefixProto, cli=False):
        clip_observations = 100.
        clip_actions = 10.0

    class obs_scales(PrefixProto, cli=False):
        lin_vel = 2.0
        ang_vel = 0.25
        dof_pos = 1.0
        dof_vel = 0.05
        imu = 0.1
        height_measurements = 5.0
        friction_measurements = 1.0
        body_height_cmd = 2.0
        gait_phase_cmd = 1.0
        gait_freq_cmd = 1.0
        footswing_height_cmd = 0.15
        body_pitch_cmd = 0.3
        body_roll_cmd = 0.3
        aux_reward_cmd = 1.0
        compliance_cmd = 1.0
        stance_width_cmd = 1.0
        stance_length_cmd = 1.0
        segmentation_image = 1.0
        rgb_image = 1.0
        depth_image = 1.0
        ball_pos = 1.0

    class noise(PrefixProto, cli=False):
        add_noise = True
        noise_level = 1.0  # scales other values

    class noise_scales(PrefixProto, cli=False):
        dof_pos = 0.01  # For JointPositionSensor
        dof_vel = 1.5   # For JointVelocitySensor
        # lin_vel = 0.1
        # ang_vel = 0.2
        # imu = 0.1
        gravity = 0.05
        feet_contact_states = 0.05
        # height_measurements = 0.1
        # friction_measurements = 0.0
        # segmentation_image = 0.0
        # rgb_image = 0.0
        # depth_image = 0.0
        ball_pos = 0.05

    # viewer camera:
    class viewer(PrefixProto, cli=False):
        ref_env = 0
        pos = None
        lookat = None
        # pos = [20, 0, 6]  # [m]
        # lookat = [11., 5, 3.]  # [m]

    class sim(PrefixProto, cli=False):
        dt = 0.005
        substeps = 1
        gravity = [0., 0., -9.81]  # [m/s^2]
        up_axis = 1  # 0 is y, 1 is z

        use_gpu_pipeline = True

        class physx_mini(PrefixProto, cli=False):
            num_threads = 10
            solver_type = 1  # 0: pgs, 1: tgs
            num_position_iterations = 4
            num_velocity_iterations = 0
            contact_offset = 0.01  # [m]
            rest_offset = 0.0  # [m]
            bounce_threshold_velocity = 0.5  # 0.5 [m/s]
            max_depenetration_velocity = 1.0
            max_gpu_contact_pairs = 2 ** 23 # 2**24 -> needed for 8000 envs and more
            default_buffer_size_multiplier = 5
            contact_collection = 2  # 0: never, 1: last sub-step, 2: all sub-steps (default=2)

        class physx_full(PrefixProto, cli=False):
            num_threads = 10
            solver_type = 1  # 0: pgs, 1: tgs
            num_position_iterations = 4
            num_velocity_iterations = 0
            contact_offset = 0.01  # [m]
            rest_offset = 0.0  # [m]
            bounce_threshold_velocity = 0.5  # 0.5 [m/s]
            max_depenetration_velocity = 1.0
            max_gpu_contact_pairs = 2 ** 24 # 2**24 -> needed for 8000 envs and more
            default_buffer_size_multiplier = 64
            contact_collection = 2  # 0: never, 1: last sub-step, 2: all sub-steps (default=2)

    class perception(PrefixProto, cli=False):
        measure_heights = False
        compute_heights = False
        measure_frictions = False
        compute_frictions = False
        measure_roughnesses = False
        compute_roughnesses = False

        camera_names = ["front", "left", "right", "bottom", "rear"]
        camera_poses = [[0.3, 0, 0], [0, 0.15, 0], [0, -0.15, 0], [0.1, 0, -0.1], [-0.2, 0, -0.1]]
        camera_rpys = [[0.0, 0, 0], [0, 0, 3.14 / 2], [0, 0, -3.14 / 2], [0, -3.14 / 2, 0],
                       [0, -3.14 / 2, 0]]
        compute_depth = False
        compute_rgb = False
        compute_segmentation = False

        image_height = 100
        image_width = 100
        image_horizontal_fov = 110.0 # 110 degrees


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

rank = int(os.getenv("LOCAL_RANK", "0"))
set_seed(0 + rank, torch_deterministic=False)