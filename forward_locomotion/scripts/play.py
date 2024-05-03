import isaacgym

assert isaacgym
import torch
import numpy as np
import argparse
import shutil
import pickle as pkl

from forward_locomotion.go1_gym.envs import *
from forward_locomotion.go1_gym.envs.base.legged_robot_config import Cfg
from forward_locomotion.go1_gym.envs.go1.go1_config import config_go1
from forward_locomotion.go1_gym.envs.mini_cheetah.velocity_tracking import VelocityTrackingEasyEnv

from tqdm import tqdm


def load_env(label, headless=False, dr_config=None, save_video=True):
    # Will be overwritten by the loaded config from parameters.pkl
    Cfg.commands = Cfg.commands_original
    Cfg.rewards = Cfg.rewards_eureka
    Cfg.domain_rand = Cfg.domain_rand_off

    # prepare environment
    config_go1(Cfg)

    from ml_logger import logger
    from forward_locomotion.go1_gym_learn.ppo.ppo import PPO_Args
    from forward_locomotion.go1_gym_learn.ppo.actor_critic import AC_Args
    from forward_locomotion.go1_gym_learn.ppo import RunnerArgs

    with open(label + "/parameters.pkl", 'rb') as file:
        pkl_cfg = pkl.load(file)
        cfg = pkl_cfg["Cfg"]
    def set_cfg_recursive(cfg, load):
        for key, value in load.items():
            if not hasattr(cfg, key):
                continue
            if key in ["commands_original", "commands_constrained", "domain_rand_original", "domain_rand_eureka", "domain_rand_off", "rewards_original", "rewards_eureka"]:
                # Don't overwrite presets from Cfg
                continue
            if isinstance(value, dict):
                set_cfg_recursive(getattr(cfg, key), value)
            else:
                setattr(cfg, key, value)
    set_cfg_recursive(Cfg, cfg)
    Cfg.commands.command_curriculum = False

    if dr_config == "original":
        Cfg.domain_rand = Cfg.domain_rand_original
    elif dr_config == "eureka":
        Cfg.domain_rand = Cfg.domain_rand_eureka
    elif dr_config == "off":
        Cfg.domain_rand = Cfg.domain_rand_off
    
    Cfg.env.record_video = save_video
    Cfg.env.num_recording_envs = 1 if save_video else 0
    Cfg.env.num_envs = 1
    Cfg.terrain.num_rows = 3
    Cfg.terrain.num_cols = 5
    Cfg.terrain.border_size = 0

    from forward_locomotion.go1_gym.envs.wrappers.history_wrapper import HistoryWrapper

    env = VelocityTrackingEasyEnv(sim_device='cuda:0', headless=headless, cfg=Cfg)
    env = HistoryWrapper(env)

    # load policy
    from ml_logger import logger
    from forward_locomotion.go1_gym_learn.ppo.actor_critic import ActorCritic

    actor_critic = ActorCritic(
        num_obs=Cfg.env.num_observations,
        num_privileged_obs=Cfg.env.num_privileged_obs,
        num_obs_history=Cfg.env.num_observations * \
                        Cfg.env.num_observation_history,
        num_actions=Cfg.env.num_actions)

    weights = logger.load_torch("checkpoints/ac_weights_last.pt")
    actor_critic.load_state_dict(state_dict=weights)
    actor_critic.to(env.device)
    policy = actor_critic.act_inference

    return env, policy


def play_mc(iterations=1000, headless=True, label=None, dr_config=None, verbose=False, save_video=False):
    from ml_logger import logger

    from pathlib import Path
    from forward_locomotion.go1_gym import MINI_GYM_ROOT_DIR
    import glob
    import os

    logger.configure(label)

    env, policy = load_env(label, headless=headless, dr_config=dr_config, save_video=save_video)

    num_eval_steps = iterations

    measured_global_x_vels = np.zeros(num_eval_steps)
    target_x_vels = np.ones(num_eval_steps) * 2.0
    joint_positions = np.zeros((num_eval_steps, 12))
    joint_velocities = np.zeros((num_eval_steps, 12))
    torques = np.zeros((num_eval_steps, 12))

    if save_video:
        import imageio
        mp4_writer = imageio.get_writer('locomotion.mp4', fps=50)

    obs = env.reset()

    starting_pos = env.root_states[0, :3].cpu().numpy()
    for i in tqdm(range(num_eval_steps)):
        env.commands[:, :] = 0.0
        env.commands[:, 0] = 2.0
        with torch.no_grad():
            actions = policy(obs)
        obs, rew, done, info = env.step(actions)
        if verbose:
            print(f'linear velocity: {info["body_global_linear_vel"]}')
            print(f"distance traveled (x): {(env.root_states[0, 0].cpu().numpy() - starting_pos)[0]}")
        measured_global_x_vels[i] = env.root_states[0, 7]
        joint_positions[i] = env.dof_pos[0, :].cpu()
        joint_velocities[i] = env.dof_vel[0, :].cpu()
        torques[i] = env.torques[0, :].detach().cpu()

        # Stop after running 5 meters
        # if (env.root_states[0, :3].cpu().numpy() - starting_pos)[0] >= 5.0:
        #     break

        if save_video:
            img = env.render(mode='rgb_array')
            mp4_writer.append_data(img)
    
    if save_video:
        mp4_writer.close()
        video_dir_path = os.path.join(label, f"{logger.prefix}/videos")
        if not os.path.exists(video_dir_path):
            os.makedirs(video_dir_path)
        shutil.move("locomotion.mp4", os.path.join(video_dir_path, "play.mp4"))
    
        np.savez(os.path.join(video_dir_path, "plot_data.npz"),
                measured_global_x_vels=measured_global_x_vels, target_x_vels=target_x_vels, joint_positions=joint_positions, joint_velocities=joint_velocities, torques=torques)

        # plot target and measured forward velocity
        from matplotlib import pyplot as plt
        fig, axs = plt.subplots(4, 1, figsize=(12, 10))
        axs[0].plot(np.linspace(0, num_eval_steps * env.dt, num_eval_steps), measured_global_x_vels, color='black', linestyle="-", label="Measured")
        axs[0].plot(np.linspace(0, num_eval_steps * env.dt, num_eval_steps), target_x_vels, color='black', linestyle="--", label="Desired")
        axs[0].legend()
        axs[0].set_title("Global Forward Linear Velocity")
        axs[0].set_xlabel("Time (s)")
        axs[0].set_ylabel("Velocity (m/s)")

        axs[1].plot(np.linspace(0, num_eval_steps * env.dt, num_eval_steps), joint_positions, linestyle="-", label="Measured")
        axs[1].set_title("Joint Positions")
        axs[1].set_xlabel("Time (s)")
        axs[1].set_ylabel("Joint Position (rad)")

        axs[2].plot(np.linspace(0, num_eval_steps * env.dt, num_eval_steps), joint_velocities, linestyle="-", label="Measured")
        axs[2].set_title("Joint Velocities")
        axs[2].set_xlabel("Time (s)")
        axs[2].set_ylabel("Joint Velocity (rad/s)")

        axs[3].plot(np.linspace(0, num_eval_steps * env.dt, num_eval_steps), torques, linestyle="-", label="Measured")
        axs[3].set_title("Joint Torques")
        axs[3].set_xlabel("Time (s)")
        axs[3].set_ylabel("Torque (Nm)")

        plt.tight_layout()
        plt.savefig(os.path.join(video_dir_path, "plot.png"))
        if not headless:
            plt.show()
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=str, default=None)
    parser.add_argument('--iterations', type=int, default=1000)
    parser.add_argument('--headless', action='store_true')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument("--dr-config", type=str, choices=["original", "eureka", "off"])
    parser.add_argument("--no-video", action="store_true")
    args = parser.parse_args()

    play_mc(iterations=args.iterations, headless=args.headless, label=args.run, dr_config=args.dr_config, verbose=args.verbose, save_video=not args.no_video)