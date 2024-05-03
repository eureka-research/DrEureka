import isaacgym

assert isaacgym
import torch
import numpy as np
import argparse
import shutil
import pickle as pkl
import os
import subprocess
import json
import hydra
from tqdm import tqdm

from utils.misc import *

DR_EUREKA_ROOT_DIR = os.getcwd()
ROOT_DIR = f"{DR_EUREKA_ROOT_DIR}/.."

def forward_locomotion_success(stdout, parameter, val):
    """How close forward velocity is to target velocity, linearly measured."""
    lines = stdout.decode().split("\n")
    lines = [line for line in lines if "linear velocity" in line]

    target_forward_vel = 2.0
    average_success, average_forward_vel, counter = 0.0, 0.0, 0
    for line in lines:
        forward_vel = float(line.split(": ")[1].strip("[]").split()[0])
        average_success += -abs(forward_vel - target_forward_vel)
        average_forward_vel += forward_vel
        counter += 1
    average_success /= counter
    average_forward_vel /= counter
    print(f"Average success and forward vel for {parameter} = {val}: {average_success}, {average_forward_vel}")
    return average_success >= -1.0

def globe_walking_success(stdout, parameter, val):
    """Whether the robot is balanced on the ball (and not floating)."""
    lines = stdout.decode().split("\n")
    lines = [line for line in lines if "z-position" in line]

    min_base_height, max_base_height = float("inf"), float("-inf")
    for line in lines:
        height = float(line.split(": ")[1].strip())
        min_base_height = min(min_base_height, height)
        max_base_height = max(max_base_height, height)
    print(f"Minimum, Maximum base height for {parameter} = {val}: {min_base_height}, {max_base_height}")
    return 0.9 <= min_base_height and max_base_height <= 1.2

def success(cfg, stdout, parameter, val):
    """
    Success criteria for a given parameter value.
    This criteria must consider extreme environment conditions, unlike the one for reward generation.
    For example, extreme gravity can cause the robot to float or move with high velocity.
    """

    if cfg.env.env_name == "forward_locomotion":
        return forward_locomotion_success(stdout, parameter, val)
    elif cfg.env.env_name == "globe_walking":
        return globe_walking_success(stdout, parameter, val)
    else:
        raise NotImplementedError

def increase_perturbation_interval(cfg, dr, parameter):
    # Increases frequency of perturbations to test within short play iterations
    if cfg.env.env_name == "forward_locomotion":
        if parameter == "gravity_range":
            dr = dr.replace("gravity_rand_interval_s = 10", "gravity_rand_interval_s = 1")
        if parameter == "push_vel_xy_range":
            dr = dr.replace("push_interval_s = 15", "push_interval_s = 1")
    elif cfg.env.env_name == "globe_walking":
        if parameter == "gravity_range":
            dr = dr.replace("gravity_rand_interval_s = 10", "gravity_rand_interval_s = 0.5")
        if parameter == "robot_push_vel_range":
            dr = dr.replace("push_robot_interval_s = 15", "push_robot_interval_s = 0.5")
        if parameter == "ball_push_vel_range":
            dr = dr.replace("push_ball_interval_s = 10", "push_ball_interval_s = 0.5")
        if parameter == "ball_drag_range":
            dr = dr.replace("drag_ball_interval_s = 15", "drag_ball_interval_s = 1")
    else:
        raise NotImplementedError
    return dr

def test_parameter(cfg, parameter, test_vals):
    env_name = cfg.env.env_name
    dr_template_file = f'{ROOT_DIR}/{env_name}/{cfg.env.dr_template_file}'
    dr_template = file_to_string(dr_template_file)
    output_file = f"{ROOT_DIR}/{env_name}/{cfg.env.dr_output_file}"

    lowest_successful_idx, highest_successful_idx = float("inf"), float("-inf")
    print(f"Testing parameter: {parameter}")
    for idx, val in enumerate(tqdm(test_vals)):
        content = f"{parameter} = [{val}, {val}]"
        dr = dr_template.replace("# INSERT EUREKA DR HERE", " "*8 + content)
        dr = increase_perturbation_interval(cfg, dr, parameter)
        with open(output_file, 'w') as f:
            f.write(dr)

        set_freest_gpu()
        command = f"python -u {ROOT_DIR}/{cfg.env.env_name}/{cfg.env.play_script} --headless --iterations {cfg.env.play_iterations} --run {cfg.run_path} --dr-config eureka --no-video --verbose"
        command = command.split(" ")
        os.environ["TQDM_DISABLE"] = "1"
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()
        # if stderr:
        #     stderr = stderr.decode()
        #     print(stderr)
        #     raise RuntimeError("Error in subprocess")

        if success(cfg, stdout, parameter, val):
            lowest_successful_idx = min(lowest_successful_idx, idx)
            highest_successful_idx = max(highest_successful_idx, idx)
    
    if lowest_successful_idx == float("inf") and highest_successful_idx == float("-inf"):
        raise RuntimeError("No successful parameter values found, please double check your model checkpoint!")
    
    return test_vals[lowest_successful_idx], test_vals[highest_successful_idx]

@hydra.main(config_path="cfg", config_name="config_rapp", version_base="1.1")
def main(cfg):
    # Generic ranges based on valid bounds
    min_0 = [0.0, 0.01, 0.1, 0.5, 1.0, 5.0, 10.0]
    limit_01 = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    centered_0 = [-10.0, -5.0, -1.0, -0.5, -0.1, -0.01, 0.0, 0.01, 0.1, 0.5, 1.0, 5.0, 10.0]
    centered_1 = [0.0, 0.5, 0.9, 1.0, 1.1, 1.5, 2.0]

    if cfg.env.env_name == "forward_locomotion":
        parameter_test_vals = {
            "friction_range": min_0,
            "restitution_range": limit_01,
            "added_mass_range": centered_0,
            "com_displacement_range": centered_0,
            "motor_strength_range": centered_1,
            "Kp_factor_range": centered_1,
            "Kd_factor_range": centered_1,
            "dof_stiffness_range": min_0,
            "dof_damping_range": min_0,
            "dof_friction_range": min_0,
            "dof_armature_range": min_0,
            "push_vel_xy_range": min_0,
            "gravity_range": centered_0,
        }
        parameter_hints = {
            "dof_armature_range": "This is the range of values added onto the diagonal of the joint inertia matrix.",
            "push_vel_xy_range": "This is the range of magnitudes of a vector added onto the robot's xy velocity.",
            "gravity_range": "This is the range of values added onto each dimension of [0.0, 0.0, -9.8]. For example, [0.0, 0.0] would keep gravity constant."
        }
    elif cfg.env.env_name == "globe_walking":
        parameter_test_vals = {
            "robot_friction_range": min_0,
            "robot_restitution_range": limit_01,
            "robot_payload_mass_range": centered_0,
            "robot_com_displacement_range": centered_0,
            "robot_motor_strength_range": centered_1,
            "robot_motor_offset_range": centered_0,

            "ball_mass_range": min_0,
            "ball_friction_range": min_0,
            "ball_restitution_range": limit_01,
            "ball_drag_range": min_0,

            "terrain_ground_friction_range": min_0,
            "terrain_ground_restitution_range": limit_01,
            "terrain_tile_roughness_range": min_0,

            "robot_push_vel_range": min_0,
            "ball_push_vel_range": min_0,
            "gravity_range": centered_0,
        }
        parameter_hints = {}
    else:
        raise NotImplementedError

    parameter_ranges = {}
    for parameter, test_vals in parameter_test_vals.items():
        parameter_ranges[parameter] = test_parameter(cfg, parameter, test_vals)
        print(parameter_ranges)

    RAPP_str = ""
    for parameter, (min_val, max_val) in parameter_ranges.items():
        print(f"{parameter} = [{min_val}, {max_val}]")
        RAPP_str += " "*4 + f"{parameter} = [{min_val}, {max_val}]"
        if parameter in parameter_hints:
            RAPP_str += " "*4 + f"({parameter_hints[parameter]})"
        RAPP_str += "\n"
    
    with open(f"{DR_EUREKA_ROOT_DIR}/prompts/initial_users/initial_users_template.txt") as f:
        template = f.read()
    output_file = f"{DR_EUREKA_ROOT_DIR}/prompts/initial_users/{cfg.env.env_name}.txt"
    if os.path.exists(output_file):
        os.rename(output_file, f"{output_file}.old")
    with open(output_file, 'w') as f:
        f.write(template.format(RAPP_bounds=RAPP_str, task_description="{task_description}"))

if __name__ == "__main__":
    main()