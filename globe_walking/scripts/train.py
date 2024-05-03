import sys
import os 
import argparse

def train_go1(iterations, dr_config, headless=True, resume_path=None, no_wandb=False, wandb_group=None):

    import isaacgym
    assert isaacgym
    import torch

    from globe_walking.go1_gym.envs.base.legged_robot_config import Cfg
    from globe_walking.go1_gym.envs.go1.go1_config import config_go1
    from globe_walking.go1_gym.envs.go1.velocity_tracking import VelocityTrackingEasyEnv

    from globe_walking.go1_gym_learn.ppo_cse import Runner
    from globe_walking.go1_gym.envs.wrappers.history_wrapper import HistoryWrapper
    from globe_walking.go1_gym_learn.ppo_cse.actor_critic import AC_Args
    from globe_walking.go1_gym_learn.ppo_cse.ppo import PPO_Args
    from globe_walking.go1_gym_learn.ppo_cse import RunnerArgs

    from ml_logger import logger

    if dr_config == "eureka":
        Cfg.env = Cfg.env_full
        Cfg.sensors = Cfg.sensors_full
        Cfg.terrain = Cfg.terrain_full
        Cfg.domain_rand = Cfg.domain_rand_eureka
        Cfg.sim.physx = Cfg.sim.physx_full
    elif dr_config == "off":
        Cfg.env = Cfg.env_mini
        Cfg.sensors = Cfg.sensors_mini
        Cfg.terrain = Cfg.terrain_mini
        Cfg.domain_rand = Cfg.domain_rand_off
        Cfg.sim.physx = Cfg.sim.physx_mini
    else:
        raise ValueError(f"Invalid dr_config: {dr_config}")

    config_go1(Cfg)

    if resume_path:
        RunnerArgs.resume = True
        RunnerArgs.load_run = resume_path
        RunnerArgs.resume_checkpoint = os.path.join(RunnerArgs.load_run, "checkpoints", "ac_weights_last.pt")

    Cfg.robot.name = "go1"

    Cfg.commands.num_lin_vel_bins = 30
    Cfg.commands.num_ang_vel_bins = 30
    Cfg.curriculum_thresholds.tracking_ang_vel = 0.7
    Cfg.curriculum_thresholds.tracking_lin_vel = 0.8
    Cfg.curriculum_thresholds.tracking_contacts_shaped_vel = 0.90
    Cfg.curriculum_thresholds.tracking_contacts_shaped_force = 0.90

    Cfg.commands.distributional_commands = True

    Cfg.control.control_type = "actuator_net"

    Cfg.env.num_observation_history = 15

    Cfg.commands.exclusive_phase_offset = False
    Cfg.commands.pacing_offset = False
    Cfg.commands.balance_gait_distribution = False
    Cfg.commands.binary_phases = False
    Cfg.commands.gaitwise_curricula = False

    ###############################
    # globe walking configuration
    ###############################

    # ball parameters
    Cfg.env.add_balls = True

    # sensory observation
    Cfg.commands.num_commands = 0
    Cfg.env.episode_length_s = 40.
    Cfg.env.num_observations = 56

    # terrain configuration
    Cfg.terrain.border_size = 0.0
    Cfg.terrain.mesh_type = "boxes_tm"
    Cfg.terrain.num_cols = 20
    Cfg.terrain.num_rows = 20
    Cfg.terrain.terrain_length = 5.0
    Cfg.terrain.terrain_width = 5.0
    Cfg.terrain.num_border_boxes = 5.0
    Cfg.terrain.teleport_thresh = 0.3
    Cfg.terrain.teleport_robots = False
    Cfg.terrain.center_robots = False
    Cfg.terrain.center_span = 3
    Cfg.terrain.horizontal_scale = 0.05
    Cfg.terrain.terrain_proportions = [1.0, 0.0, 0.0, 0.0, 0.0]
    Cfg.terrain.curriculum = False
    Cfg.terrain.difficulty_scale = 1.0
    Cfg.terrain.max_step_height = 0.26
    Cfg.terrain.min_step_run = 0.25
    Cfg.terrain.max_step_run = 0.4
    Cfg.terrain.max_init_terrain_level = 1

    # terminal conditions
    Cfg.rewards.use_terminal_body_height = True
    Cfg.rewards.use_terminal_roll_pitch = False
    Cfg.rewards.reward_container_name = "EurekaReward"
    Cfg.asset.terminate_after_contacts_on = []

    AC_Args.adaptation_labels = []
    AC_Args.adaptation_dims = []

    RunnerArgs.save_video_interval = 500

    import wandb
    if (Cfg.multi_gpu and int(os.getenv("LOCAL_RANK", "0")) == 0) or not Cfg.multi_gpu:
        time_now = logger.utcnow(f'globe_walking/%Y-%m-%d/{Path(__file__).stem}/%H%M%S.%f')
        logger.configure(time_now, root=Path(f"{MINI_GYM_ROOT_DIR}/runs").resolve(), )
        logger.log_text("""
                    charts: 
                    - yKey: train/episode/rew_total/mean
                    xKey: iterations
                    - yKey: train/episode/rew_tracking_lin_vel/mean
                    xKey: iterations
                    - yKey: train/episode/rew_tracking_contacts_shaped_force/mean
                    xKey: iterations
                    - yKey: train/episode/rew_action_smoothness_1/mean
                    xKey: iterations
                    - yKey: train/episode/rew_action_smoothness_2/mean
                    xKey: iterations
                    - yKey: train/episode/rew_tracking_contacts_shaped_vel/mean
                    xKey: iterations
                    - yKey: train/episode/rew_orientation_control/mean
                    xKey: iterations
                    - yKey: train/episode/rew_dof_pos/mean
                    xKey: iterations
                    - yKey: train/episode/command_area_trot/mean
                    xKey: iterations
                    - yKey: train/episode/max_terrain_height/mean
                    xKey: iterations
                    - type: video
                    glob: "videos/*.mp4"
                    - yKey: adaptation_loss/mean
                    xKey: iterations
                    """, filename=".charts.yml", dedent=True)
        logger.log_params(AC_Args=vars(AC_Args), PPO_Args=vars(PPO_Args), RunnerArgs=vars(RunnerArgs),
                        Cfg=vars(Cfg))

        run_name = logger.prefix.split("/")[-1]
        name_prefix = wandb_group + "/" if wandb_group is not None else ""
        wandb.init(
            project="globe_walking",
            entity="upenn-pal",
            name=f"{name_prefix}{run_name}",
            group=wandb_group,
            config={
                "AC_Args": vars(AC_Args),
                "PPO_Args": vars(PPO_Args),
                "RunnerArgs": vars(RunnerArgs),
                "Cfg": vars(Cfg),
            },
            mode=("disabled" if no_wandb else "online")
        )


    if Cfg.multi_gpu:
        rank = int(os.getenv("LOCAL_RANK", "0"))
        device = f'cuda:{rank}'
    else:
        device = 'cuda:0'
    env = VelocityTrackingEasyEnv(sim_device=device, headless=headless, cfg=Cfg)

    env = HistoryWrapper(env)
    runner = Runner(env, device=device, multi_gpu=Cfg.multi_gpu)
    runner.learn(num_learning_iterations=int(iterations), init_at_random_ep_len=True, eval_freq=100)


if __name__ == '__main__':
    from pathlib import Path
    from globe_walking.go1_gym import MINI_GYM_ROOT_DIR
    from ml_logger import logger

    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=50000)
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--wandb-group", type=str)

    parser.add_argument("--dr-config", type=str, required=True, choices=["eureka", "off"])
    parser.add_argument("--reward-config", type=str, required=True, choices=["eureka"])
    args = parser.parse_args()

    assert args.reward_config == "eureka", "Only Eureka reward is available"

    resume_path = None
    train_go1(iterations=args.iterations, dr_config=args.dr_config, headless=True, resume_path=resume_path, no_wandb=args.no_wandb, wandb_group=args.wandb_group)

