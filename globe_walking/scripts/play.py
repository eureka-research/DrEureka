import isaacgym
assert isaacgym

import argparse
import sys
import torch
import numpy as np
import glob
import pickle as pkl
import os
import yaml
import shutil
import wandb

from globe_walking.go1_gym.envs import *
from globe_walking.go1_gym.envs.base.legged_robot_config import Cfg
from globe_walking.go1_gym.envs.go1.go1_config import config_go1
from globe_walking.go1_gym.envs.go1.velocity_tracking import VelocityTrackingEasyEnv

from tqdm import tqdm

def load_policy(label):
    body = torch.jit.load(label + '/body_latest.jit', map_location="cpu")
    adaptation_module = torch.jit.load(label + '/adaptation_module_latest.jit', map_location='cpu')

    # Alternative loading method using wandb
    if False:
        body_file = wandb.restore('tmp/legged_data/body_68000.jit', run_path=label)
        body = torch.jit.load(body_file.name)
        adaptation_module_file = wandb.restore('tmp/legged_data/adaptation_module_68000.jit', run_path=label)
        adaptation_module = torch.jit.load(adaptation_module_file.name)

    def policy(obs, info={}):
        i = 0
        latent = adaptation_module.forward(obs["obs_history"].to('cpu'))
        action = body.forward(torch.cat((obs["obs_history"].to('cpu'), latent), dim=-1))
        info['latent'] = latent
        return action

    return policy


def load_env(label, headless=False, dr_config="off", save_video=True):
    # Will be overwritten by the loaded config from parameters.pkl
    Cfg.env = Cfg.env_mini
    Cfg.sensors = Cfg.sensors_mini
    Cfg.terrain = Cfg.terrain_mini
    Cfg.domain_rand = Cfg.domain_rand_off
    Cfg.sim.physx = Cfg.sim.physx_mini

    config_go1(Cfg)
    if os.path.exists(label + "/config.yaml"):
        with open(label + "/config.yaml", 'rb') as file: 
            cfg = yaml.safe_load(file)
            cfg = cfg["Cfg"]
    elif os.path.exists(label + "/../parameters.pkl"):
        with open(label + "/../parameters.pkl", 'rb') as file:
            pkl_cfg = pkl.load(file)
            cfg = pkl_cfg["Cfg"]
    else:
        # Load from wandb, currently unused and expected to fail
        cfg_file = wandb.restore('config.yaml', run_path=label)
        with open(cfg_file.name, 'rb') as file: 
            cfg = yaml.safe_load(file)
            cfg = cfg["Cfg"]["value"]
        
    def set_cfg_recursive(cfg, load):
        for key, value in load.items():
            if not hasattr(cfg, key):
                continue
            if dr_config != "load" and key in ["env_mini", "env_full", "sensors_mini", "sensors_full", "terrain_mini", "terrain_full", "domain_rand_mini", "domain_rand_full", "domain_rand_eureka", "physx_mini", "physx_full"]:
                # Don't overwrite presets from Cfg
                continue
            if key in ["pos", "ball_init_pos"]:
                # Backwards compatibility
                continue
            if isinstance(value, dict):
                set_cfg_recursive(getattr(cfg, key), value)
            else:
                # if value != getattr(cfg, key):
                #     print(f"Overwriting {key} from {getattr(cfg, key)} to {value}")
                setattr(cfg, key, value)
    set_cfg_recursive(Cfg, cfg)
    Cfg.multi_gpu = False

    if dr_config == "eureka":
        Cfg.domain_rand = Cfg.domain_rand_eureka
    elif dr_config == "off":
        Cfg.domain_rand = Cfg.domain_rand_off
    elif dr_config == "load":
        pass  # Load from the loaded config
    else:
        raise ValueError("Invalid domain randomization configuration")
    Cfg.domain_rand.randomize = False

    Cfg.env.record_video = save_video

    Cfg.env.num_recording_envs = 1
    Cfg.env.num_envs = 1
    Cfg.terrain.num_rows = 5
    Cfg.terrain.num_cols = 5
    Cfg.terrain.border_size = 0
    Cfg.terrain.num_border_boxes = 0
    Cfg.terrain.center_robots = True
    Cfg.terrain.center_span = 1
    Cfg.terrain.teleport_robots = True

    Cfg.control.control_type = "actuator_net"

    # The following are a series of tests to verify that DR is working as expected
    if False:
        # For visualizing multiple envs
        Cfg.env.num_envs = 3
        Cfg.terrain.center_robots = False
    if False:
        # Put quadruped on ground, get rid of ball
        Cfg.domain_rand.ball_radius_range = [0.0, 0.0]
    if False:
        # Put the ball somewhere fall away
        Cfg.ball.init_pos_range = [100.0, 100.0, 0.2]
    if False:
        # Drop quadruped and ball (to test restitution)
        Cfg.ball.ball_init_pos = [1.0, 1.0, 1.0]
        Cfg.init_state.pos[-1] = 10.0

    # Extreme values for testing
    if False:
        print("> Friction Test: quadruped should slip off easily")
        Cfg.domain_rand.robot_friction_range = [0.0, 0.0]
        Cfg.domain_rand.ball_friction_range = [0.0, 0.0]
    if False:
        print("> Mass test: ball should be immovable")
        Cfg.domain_rand.ball_mass_range = [1000.0, 1000.0]
        Cfg.domain_rand.terrain_tile_roughness_range = [0.0, 0.0]     # Disable terrain so balls don't move due to gravity
        Cfg.domain_rand.ball_push_vel_range = [0.0, 0.0]
        Cfg.domain_rand.gravity_range = [0.0, 0.0]
    if False:
        print("> Radius test: balls should be big and vary in size, quadrupeds should spawn perfectly on top")
        Cfg.domain_rand.ball_radius_range = [0.0, 2.0]
    if False:
        print("> Restitution test 1: UNSTABLE IN PUBLIC ISAACGYM")
        # To make the effect clear, set Cfg.ball.ball_init_pos = [1.0, 1.0, 1.0] as well
        Cfg.domain_rand.robot_restitution_range = [10.0, 10.0]
        Cfg.domain_rand.ball_restitution_range = [1.0, 1.0]
        Cfg.domain_rand.ball_compliance_range = [0.0, 0.0]
        Cfg.domain_rand.ball_drag_range = [0.0, 0.0]
        Cfg.domain_rand.ball_push_vel_range = [0.0, 0.0]
        Cfg.domain_rand.gravity_range = [0.0, 0.0]
        Cfg.domain_rand.terrain_tile_roughness_range = [0.0, 0.0]

    if False:
        print("> Restitution test 2: UNSTABLE IN PUBLIC ISAACGYM")
        ball_restitution_range = [0.0, 0.0]
        Cfg.domain_rand.terrain_ground_restitution_range = [0.0, 0.0]
    if False: 
        print("> Compliance test: NOT IMPLEMENTED IN PUBLIC ISAACGYM")
        Cfg.domain_rand.ball_compliance_range = [10.0, 10.0]
    if False:
        print("> Drag test: ball should move less")
        Cfg.domain_rand.ball_drag_range = [500.0, 500.0]
    if False:
        print("> Push test: quadruped and ball should get pushed around violently")
        Cfg.domain_rand.push_robot_interval_s = 1
        Cfg.domain_rand.robot_push_vel_range = [10.0, 10.0]
        Cfg.domain_rand.push_ball_interval_s = 1
        Cfg.domain_rand.ball_push_vel_range = [10.0, 10.0]
    if False:
        print("> Gravity test: quadrupeds and balls should shift around")
        Cfg.domain_rand.gravity_range = [-3.0, 3.0]
        Cfg.domain_rand.gravity_rand_interval_s = 1
        Cfg.terrain.x_init_range = 0.0
        Cfg.terrain.y_init_range = 0.0
    if False:
        print("> Payload test: quadruped leg should be more bent, unable to support itself")
        Cfg.domain_rand.robot_payload_mass_range = [10.0, 10.0]
    if False:
        print("> CoM test: quadruped should tilt to one side")
        Cfg.domain_rand.robot_com_displacement_range = [0.5, 0.5]
        Cfg.terrain.x_init_range = 0.0
        Cfg.terrain.y_init_range = 0.0
    if False:
        print("> Inertia test: the ball should be harder to rotate")
        Cfg.domain_rand.ball_inertia_multiplier_range = [1000.0, 1000.0]
    if False:
        print("> Spring coefficient test: robot's feet should bounce off the ball")
        Cfg.domain_rand.ball_spring_coefficient_range = [0.7, 0.7]

    from globe_walking.go1_gym.envs.wrappers.history_wrapper import HistoryWrapper

    env = VelocityTrackingEasyEnv(sim_device='cuda:0', headless=headless, cfg=Cfg)
    env = HistoryWrapper(env)

    policy = load_policy(label)
    return env, policy


def play_go1(iterations, headless=True, label=None, dr_config="off", verbose=False, save_video=False):
    label = os.path.join(label, "checkpoints")
    env, policy = load_env(label, headless=headless, dr_config=dr_config)

    measured_x_vels = np.zeros(iterations)
    measured_global_x_vels = np.zeros(iterations)
    joint_positions = np.zeros((iterations, 12))
    foot_contact_forces = np.zeros((iterations, 4))
    torques = np.zeros((iterations, 12))

    if save_video:
        import imageio
        mp4_writer = imageio.get_writer('globe_walking.mp4', fps=50)

    obs = env.reset()
    ep_rew = 0
    for i in tqdm(range(iterations)):
        with torch.no_grad():
            actions = policy(obs)
        obs, rew, done, info = env.step(actions)
        if verbose:
            print(f"z-position: {env.base_pos[0, 2]}")
        measured_x_vels[i] = env.base_lin_vel[0, 0]
        measured_global_x_vels[i] = env.root_states[0, 7]
        joint_positions[i] = env.dof_pos[0, :].cpu()
        foot_contact_forces[i] = torch.norm(env.contact_forces[0, env.feet_indices, :], dim=-1).cpu()
        torques[i] = env.torques[0, :].detach().cpu()
        ep_rew += rew

        if save_video:
            img = env.render(mode='rgb_array')
            mp4_writer.append_data(img)

        out_of_limits = -(env.dof_pos - env.dof_pos_limits[:, 0]).clip(max=0.)
        out_of_limits += (env.dof_pos - env.dof_pos_limits[:, 1]).clip(min=0.)

    if save_video:
        mp4_writer.close()
        video_dir_path = os.path.join(label, "../videos")
        if not os.path.exists(video_dir_path):
            os.makedirs(video_dir_path)
        shutil.move("globe_walking.mp4", os.path.join(video_dir_path, "play.mp4"))

        # plot target and measured forward velocity
        np.savez(os.path.join(video_dir_path, "plot_data.npz"),
                measured_x_vels=measured_x_vels, joint_positions=joint_positions, foot_contact_forces=foot_contact_forces, torques=torques)

        from matplotlib import pyplot as plt
        fig, axs = plt.subplots(5, 1, figsize=(12, 12))
        axs[0].plot(np.linspace(0, iterations * env.dt, iterations), measured_x_vels, color='black', linestyle="-", label="Measured")
        axs[0].legend()
        axs[0].set_title("Forward Linear Velocity")
        axs[0].set_xlabel("Time (s)")
        axs[0].set_ylabel("Velocity (m/s)")

        axs[1].plot(np.linspace(0, iterations * env.dt, iterations), measured_global_x_vels, color='black', linestyle="-", label="Measured")
        axs[1].legend()
        axs[1].set_title("Global Forward Linear Velocity")
        axs[1].set_xlabel("Time (s)")
        axs[1].set_ylabel("Velocity (m/s)")

        axs[2].plot(np.linspace(0, iterations * env.dt, iterations), joint_positions, linestyle="-", label="Measured")
        axs[2].set_title("Joint Positions")
        axs[2].set_xlabel("Time (s)")
        axs[2].set_ylabel("Joint Position (rad)")

        axs[3].plot(np.linspace(0, iterations * env.dt, iterations), foot_contact_forces, linestyle="-", label="Measured")
        axs[3].set_title("Foot Contact Forces")
        axs[3].set_xlabel("Time (s)")
        axs[3].set_ylabel("Force (N)")

        axs[4].plot(np.linspace(0, iterations * env.dt, iterations), torques, linestyle="-", label="Measured")
        axs[4].set_title("Torques")
        axs[4].set_xlabel("Time (s)")
        axs[4].set_ylabel("Torque (Nm)")
        axs[4].legend(env.dof_names)

        plt.tight_layout()

        plt.savefig(os.path.join(video_dir_path, "plot.png"))
        if not headless:
            plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=str, default=None)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument('--headless', action='store_true')
    parser.add_argument("--dr-config", type=str, required=True, choices=["mini", "full", "eureka", "off", "load"])
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--no-video", action="store_true")
    args = parser.parse_args()

    play_go1(iterations=args.iterations, headless=args.headless, label=args.run, dr_config=args.dr_config, verbose=args.verbose, save_video=not args.no_video)
