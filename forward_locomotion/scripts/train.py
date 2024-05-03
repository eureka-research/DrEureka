import argparse

def train_mc(iterations, command_config, reward_config, dr_config, eureka_target_velocity=None,
             headless=True, no_wandb=False, wandb_group=None, wandb_prefix=None, seed=0):

    import os
    import shutil
    import isaacgym
    assert isaacgym
    import wandb
    from ml_logger import logger

    from forward_locomotion.go1_gym.envs.base.legged_robot_config import Cfg, set_seed
    from forward_locomotion.go1_gym.envs.go1.go1_config import config_go1
    from forward_locomotion.go1_gym.envs.mini_cheetah.velocity_tracking import VelocityTrackingEasyEnv
    from forward_locomotion.go1_gym_learn.ppo import Runner
    from forward_locomotion.go1_gym.envs.wrappers.history_wrapper import HistoryWrapper
    from forward_locomotion.go1_gym_learn.ppo.actor_critic import AC_Args
    from forward_locomotion.go1_gym_learn.ppo.ppo import PPO_Args
    from forward_locomotion.go1_gym_learn.ppo import RunnerArgs

    set_seed(seed, torch_deterministic=False)

    if command_config == "original":
        Cfg.commands = Cfg.commands_original
        assert reward_config == "original"
    elif command_config == "constrained":
        Cfg.commands = Cfg.commands_constrained
        assert reward_config == "original"
    elif command_config == "off":
        Cfg.commands = Cfg.commands_original  # Will be turned off below
    else:
        raise NotImplementedError
    if reward_config == "original":
        Cfg.rewards = Cfg.rewards_original
        assert eureka_target_velocity is None
    elif reward_config == "eureka":
        Cfg.rewards = Cfg.rewards_eureka
        if eureka_target_velocity is not None:
            Cfg.rewards.target_velocity = eureka_target_velocity
    else:
        raise NotImplementedError
    if dr_config == "original":
        Cfg.domain_rand = Cfg.domain_rand_original
    elif dr_config == "eureka":
        Cfg.domain_rand = Cfg.domain_rand_eureka
    elif dr_config == "off":
        Cfg.domain_rand = Cfg.domain_rand_off
    else:
        raise NotImplementedError
    
    config_go1(Cfg)
    if command_config == "original" or command_config == "constrained":
        Cfg.commands.command_curriculum = True
        Cfg.env.observe_command = True
        Cfg.env.num_observations = 42
    else:
        Cfg.commands.command_curriculum = False

    env = VelocityTrackingEasyEnv(sim_device='cuda:0', headless=headless, cfg=Cfg)

    logger.log_params(AC_Args=vars(AC_Args), PPO_Args=vars(PPO_Args), RunnerArgs=vars(RunnerArgs),
                      Cfg=vars(Cfg))
    run_name = logger.prefix.split("/")[-1]
    name_prefix = wandb_group + "/" if wandb_group is not None else ""
    name_prefix = name_prefix + wandb_prefix + "_" if wandb_prefix is not None else name_prefix
    wandb.init(
        project="forward_locomotion",
        entity="upenn-pal",
        name=f"{name_prefix}{run_name}",
        group=wandb_group,
        config={
            "AC_Args": vars(AC_Args),
            "PPO_Args": vars(PPO_Args),
            "RunnerArgs": vars(RunnerArgs),
            "Cfg": vars(Cfg),
        },
        mode="disabled" if no_wandb else "online",
    )

    env = HistoryWrapper(env)
    gpu_id = 0
    runner = Runner(env, device=f"cuda:{gpu_id}")
    runner.learn(num_learning_iterations=iterations, init_at_random_ep_len=True, eval_freq=100)


if __name__ == '__main__':
    from pathlib import Path
    from ml_logger import logger
    from forward_locomotion.go1_gym import MINI_GYM_ROOT_DIR

    stem = Path(__file__).stem
    logger.configure(logger.utcnow(f'forward_locomotion/%Y-%m-%d/{stem}/%H%M%S.%f'),
                     root=Path(f"{MINI_GYM_ROOT_DIR}/runs").resolve(), )
    logger.log_text("""
                charts: 
                - yKey: train/episode/rew_total/mean
                  xKey: iterations
                - yKey: train/episode/command_area/mean
                  xKey: iterations
                - type: video
                  glob: "videos/*.mp4"
                """, filename=".charts.yml", dedent=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--wandb-group", type=str)
    parser.add_argument("--wandb-prefix", type=str)

    parser.add_argument("--command-config", type=str, default="off", choices=["original", "constrained", "off"])
    parser.add_argument("--reward-config", type=str, required=True, choices=["original", "eureka"])
    parser.add_argument("--dr-config", type=str, required=True, choices=["original", "eureka", "off"])

    parser.add_argument("--eureka-target-velocity", type=float)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    train_mc(iterations=args.iterations, command_config=args.command_config, reward_config=args.reward_config, dr_config=args.dr_config, eureka_target_velocity=args.eureka_target_velocity,
              headless=True, no_wandb=args.no_wandb, wandb_group=args.wandb_group, wandb_prefix=args.wandb_prefix, seed=args.seed)