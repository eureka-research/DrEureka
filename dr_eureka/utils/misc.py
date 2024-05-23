import subprocess
import os
import json
import logging

from utils.extract_task_code import file_to_string

def set_freest_gpu():
    freest_gpu = get_freest_gpu()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(freest_gpu)

def get_freest_gpu():
    # Note: if this line breaks, you can provide an absolute path to gpustat instead
    sp = subprocess.Popen(['gpustat', '--json'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out_str, _ = sp.communicate()
    gpustats = json.loads(out_str.decode('utf-8'))
    freest_gpu = min(gpustats['gpus'], key=lambda x: x['memory.used'])
    return freest_gpu['index']

def filter_traceback(s):
    lines = s.split('\n')
    filtered_lines = []
    for i, line in enumerate(lines):
        if line.startswith('Traceback'):
            for j in range(i, len(lines)):
                if "Set the environment variable HYDRA_FULL_ERROR=1" in lines[j]:
                    break
                filtered_lines.append(lines[j])
            return '\n'.join(filtered_lines)
    return ''  # Return an empty string if no Traceback is found

def block_until_training(rl_filepath, success_keyword, failure_keyword, log_status=False, iter_num=-1, response_id=-1):
    # Ensure that the RL training has started before moving on
    while True:
        rl_log = file_to_string(rl_filepath)
        if success_keyword in rl_log or failure_keyword in rl_log:
            if log_status and success_keyword in rl_log:
                if iter_num is not None:
                    logging.info(f"Iteration {iter_num}: Code Run {response_id} successfully training!")
                else:
                    logging.info(f"Code Run {response_id} successfully training!")
            if log_status and failure_keyword in rl_log:
                if iter_num is not None:
                    logging.info(f"Iteration {iter_num}: Code Run {response_id} execution error!")
                else:
                    logging.info(f"Code Run {response_id} execution error!")
            break

def construct_run_log(stdout_str):
    run_log = {}
    lines = stdout_str.split('\n')
    for i, line in enumerate(lines):
        if line.startswith("│") and line.endswith("│"):
            line = line[1:-1].split("│")
            key, val = line[0].strip(), line[1].strip()
            if key == "train/episode/rew success/mean":
                key = "consecutive_successes"
            elif key == "timesteps" or key == "iterations":
                key = key + "/"
            elif "train/episode/rew" in key:
                key = key.split("/")[2]
            elif key == "train/episode/episode length/mean":
                key = "episode length"

            run_log[key] = run_log.get(key, []) + [float(val)]
    run_log["gpt_reward"] = []
    run_log["gt_reward"] = []
    for i in range(len(run_log["consecutive_successes"])):
        cur_sum = 0
        for key in run_log:
            if "rew " in key:
                cur_sum += run_log[key][i]
        run_log["gpt_reward"].append(cur_sum)
        run_log["gt_reward"].append(cur_sum)
    return run_log