import glob
import pickle as pkl
import lcm
import sys
import os

from go1_gym_deploy.utils.deployment_runner import DeploymentRunner, NextPolicyException, PrevPolicyException
from go1_gym_deploy.envs.lcm_agent import LCMAgent
from go1_gym_deploy.utils.cheetah_state_estimator import StateEstimator
from go1_gym_deploy.utils.command_profile import *

import pathlib

lc = lcm.LCM("udpm://239.255.76.67:7667?ttl=255")

def run_policy(label, se, max_vel, max_yaw_vel):
    control_dt = 0.02
    command_profile = RCControllerProfile(dt=control_dt, state_estimator=se, x_scale=max_vel, y_scale=0.6, yaw_scale=max_yaw_vel)

    logdir = label
    print(f"Running policy: {logdir}")
    try:
        with open(logdir+"/parameters.pkl", 'rb') as file:
            pkl_cfg = pkl.load(file)
            cfg = pkl_cfg["Cfg"]
    except:
        raise Exception("Failed to load parameters.pkl")

    from go1_gym_deploy.envs.history_wrapper import HistoryWrapper
    hardware_agent = LCMAgent(cfg, se, command_profile)
    hardware_agent = HistoryWrapper(hardware_agent)

    policy = load_policy(logdir)

    # load runner
    root = f"{pathlib.Path(__file__).parent.resolve()}/../../logs/"
    pathlib.Path(root).mkdir(parents=True, exist_ok=True)
    log_prefix = "_".join(label.split("/")[-2:])
    deployment_runner = DeploymentRunner(experiment_name=experiment_name, se=None,
                                        log_root=os.path.join(root, experiment_name), log_prefix=log_prefix)
    deployment_runner.add_control_agent(hardware_agent, "hardware_closed_loop")
    deployment_runner.add_policy(policy)
    deployment_runner.add_command_profile(command_profile)

    if len(sys.argv) >= 2:
        max_steps = int(sys.argv[1])
    else:
        max_steps = 10000000
    # print(f'max steps {max_steps}')

    deployment_runner.run(max_steps=max_steps, logging=True)

def run(labels, experiment_name, max_vel=1.0, max_yaw_vel=1.0):
    se = StateEstimator(lc)
    se.spin()

    idx = 0

    while True:
        try:
            print()
            print(f"Running policy idx {idx}")
            run_policy(labels[idx], se, max_vel, max_yaw_vel)
        except NextPolicyException:
            idx = (idx + 1) % len(labels)
        except PrevPolicyException:
            idx = (idx - 1) % len(labels)

def load_policy(logdir):
    body = torch.jit.load(logdir + '/checkpoints/body_latest.jit')
    import os
    adaptation_module = torch.jit.load(logdir + '/checkpoints/adaptation_module_latest.jit')

    def policy(obs, info):
        i = 0
        latent = adaptation_module.forward(obs["obs_history"].to('cpu'))
        action = body.forward(torch.cat((obs["obs"].to('cpu'), latent), dim=-1))
        info['latent'] = latent
        return action

    return policy


if __name__ == '__main__':
    labels = [
        # Put paths to your policy here
        "../../runs/forward_locomotion/dr_eureka_best"
    ]

    experiment_name = "example_experiment"

    run(labels, experiment_name=experiment_name, max_vel=3.5, max_yaw_vel=5.0)
