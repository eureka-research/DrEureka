import numpy as np
import base64
import io
from PIL import Image
import hydra
import logging
import openai
import os
import time
from pathlib import Path
import shutil
from matplotlib import pyplot as plt
import pickle as pkl
import json

from utils.misc import * 
from utils.create_task import create_task
from utils.extract_task_code import *

EUREKA_ROOT_DIR = os.getcwd()
ROOT_DIR = f"{EUREKA_ROOT_DIR}/.."

@hydra.main(config_path="cfg", config_name="config", version_base="1.1")
def main(cfg):
    workspace_dir = Path.cwd()
    logging.info(f"Workspace: {workspace_dir}")
    logging.info(f"Project Root: {EUREKA_ROOT_DIR}")

    openai.api_key = os.getenv("OPENAI_API_KEY")

    task = cfg.env.task
    task_description = cfg.env.description
    model = cfg.model
    logging.info(f"Using LLM: {model}")
    logging.info("Task: " + task)
    logging.info("Task description: " + task_description)

    env_name = cfg.env.env_name
    dr_template_file = f'{ROOT_DIR}/{env_name}/{cfg.env.dr_template_file}'
    dr_template = file_to_string(dr_template_file)
    output_file = f"{ROOT_DIR}/{env_name}/{cfg.env.dr_output_file}"

    # Loading all text prompts
    prompt_dir = f'{EUREKA_ROOT_DIR}/prompts'
    initial_system = file_to_string(f'{prompt_dir}/initial_system.txt')
    initial_user = file_to_string(f'{prompt_dir}/initial_users/{cfg.env.env_name}.txt')
    initial_user = initial_user.format(task_description=task_description)
    messages = [{"role": "system", "content": initial_system}, {"role": "user", "content": initial_user}]

    DUMMY_FAILURE = -10000.
    max_successes = []
    max_successes_reward_correlation = []
    execute_rates = []
    best_code_paths = []
    max_success_overall = DUMMY_FAILURE
    max_success_reward_correlation_overall = DUMMY_FAILURE
    max_reward_code_path = None 
    
    # Get Eureka response
    responses = []
    response_cur = None
    total_samples = 0
    total_token = 0
    total_completion_token = 0
    chunk_size = cfg.sample if "gpt-3.5" in model else 4

    logging.info(f"Generating {cfg.sample} samples with {cfg.model}")

    while True:
        if total_samples >= cfg.sample:
            break
        for attempt in range(3):
            try:
                response_cur = openai.ChatCompletion.create(
                    model=model,
                    messages=messages,
                    temperature=cfg.temperature,
                    n=chunk_size
                )
                total_samples += chunk_size
                break
            except Exception as e:
                if attempt >= 10:
                    chunk_size = max(int(chunk_size / 2), 1)
                    print("Current Chunk Size", chunk_size)
                logging.info(f"Attempt {attempt+1} failed with error: {e}")
                time.sleep(1)
        if response_cur is None:
            logging.info("Code terminated due to too many failed attempts!")
            exit()

        responses.extend(response_cur["choices"])
        prompt_tokens = response_cur["usage"]["prompt_tokens"]
        total_completion_token += response_cur["usage"]["completion_tokens"]
        total_token += response_cur["usage"]["total_tokens"]

    if cfg.sample == 1:
        logging.info(f"GPT Output:\n " + responses[0]["message"]["content"] + "\n")

    # Logging Token Information
    logging.info(f"Prompt Tokens: {prompt_tokens}, Completion Tokens: {total_completion_token}, Total Tokens: {total_token}")

    code_runs = [] 
    rl_runs = []
    for response_id in range(cfg.sample):
        response_cur = responses[response_id]["message"]["content"]
        logging.info(f"Processing Code Run {response_id}")

        # Regex patterns to extract python code enclosed in GPT response
        patterns = [
            r'```python(.*?)```',
            r'```(.*?)```',
            r'"""(.*?)"""',
            r'""(.*?)""',
            r'"(.*?)"',
        ]
        for pattern in patterns:
            code_string = re.search(pattern, response_cur, re.DOTALL)
            if code_string is not None:
                code_string = code_string.group(1).strip()
                break
        code_string = response_cur if not code_string else code_string
        code_string = "\n".join([" "*8 + line for line in code_string.split('\n')])

        code_runs.append(code_string)
                
        # Add the Eureka Reward Signature to the environment code
        cur_task_rew_code_string = dr_template.replace("# INSERT EUREKA DR HERE", code_string)

        # Save the new environment code when the output contains valid code string!
        with open(output_file, 'w') as file:
            file.writelines(cur_task_rew_code_string + '\n')

        with open(f"config_response{response_id}_dr_only.py", 'w') as file:
            file.writelines(code_string + '\n')

        # Copy the generated environment code to hydra output directory for bookkeeping
        shutil.copy(output_file, f"config_response{response_id}.py")

        # Find the freest GPU to run GPU-accelerated RL
        set_freest_gpu()
        
        # Execute the python file with flags
        rl_filepath = f"config_response{response_id}.txt"
        with open(rl_filepath, 'w') as f:
            command = f"python -u {ROOT_DIR}/{env_name}/{cfg.env.train_script} --iterations {cfg.env.train_iterations} --dr-config eureka --reward-config eureka"
            command = command.split(" ")
            if not cfg.use_wandb:
                command.append("--no-wandb")
            process = subprocess.Popen(command, stdout=f, stderr=f)
        block_until_training(rl_filepath, success_keyword=cfg.env.success_keyword, failure_keyword=cfg.env.failure_keyword,
                                log_status=True, iter_num=None, response_id=response_id)
        rl_runs.append(process)

    # Gather RL training results and construct reward reflection
    contents = []
    successes = []
    reward_correlations = []
    code_paths = []
    
    exec_success = False 
    for response_id, (code_run, rl_run) in enumerate(zip(code_runs, rl_runs)):
        rl_run.communicate()
        rl_filepath = f"config_response{response_id}.txt"
        code_paths.append(f"config_response{response_id}.py")
        try:
            with open(rl_filepath, 'r') as f:
                stdout_str = f.read() 
        except: 
            content = "Code Run cannot be executed due to response parsing error!"
            contents.append(content) 
            successes.append(DUMMY_FAILURE)
            reward_correlations.append(DUMMY_FAILURE)
            continue

        content = ''
        traceback_msg = filter_traceback(stdout_str)

        if traceback_msg == '':
            # If RL execution has no error, provide policy statistics feedback
            exec_success = True
            run_log = construct_run_log(stdout_str)
            
            train_iterations = np.array(run_log['iterations/']).shape[0]
            epoch_freq = max(int(train_iterations // 10), 1)
            
            # Compute Correlation between Human-Engineered and GPT Rewards
            if "gt_reward" in run_log and "gpt_reward" in run_log:
                gt_reward = np.array(run_log["gt_reward"])
                gpt_reward = np.array(run_log["gpt_reward"])
                reward_correlation = np.corrcoef(gt_reward, gpt_reward)[0, 1]
                reward_correlations.append(reward_correlation)

            # Add reward components log to the feedback
            for metric in sorted(run_log.keys()):
                if "/" not in metric:
                    metric_cur = ['{:.2f}'.format(x) for x in run_log[metric][::epoch_freq]]
                    metric_cur_max = max(run_log[metric])
                    metric_cur_mean = sum(run_log[metric]) / len(run_log[metric])
                    if "consecutive_successes" == metric:
                        successes.append(metric_cur_max)
                    metric_cur_min = min(run_log[metric])
                    if metric != "gt_reward" and metric != "gpt_reward":
                        if metric != "consecutive_successes":
                            metric_name = metric 
                        else:
                            metric_name = "task score"
                        content += f"{metric_name}: {metric_cur}, Max: {metric_cur_max:.2f}, Mean: {metric_cur_mean:.2f}, Min: {metric_cur_min:.2f} \n"                    
                    else:
                        # Provide ground-truth score when success rate not applicable
                        if "consecutive_successes" not in run_log:
                            content += f"ground-truth score: {metric_cur}, Max: {metric_cur_max:.2f}, Mean: {metric_cur_mean:.2f}, Min: {metric_cur_min:.2f} \n"                    
            logging.info(f"Code Run {response_id}, Success Score: {successes[-1]:.2f}")
        else:
            # Otherwise, provide execution traceback error feedback
            successes.append(DUMMY_FAILURE)
            reward_correlations.append(DUMMY_FAILURE)
            content += f"Executing the provided domain randomization failed with the following error:\n{traceback_msg}\n"
        contents.append(content) 

    # Select the best code sample based on the success rate
    best_sample_idx = np.argmax(np.array(successes))
    best_content = contents[best_sample_idx]
        
    max_success = successes[best_sample_idx]
    max_success_reward_correlation = reward_correlations[best_sample_idx]
    execute_rate = np.sum(np.array(successes) >= 0.) / cfg.sample

    # Update the best Eureka Output
    if max_success > max_success_overall:
        max_success_overall = max_success

    execute_rates.append(execute_rate)
    max_successes.append(max_success)
    max_successes_reward_correlation.append(max_success_reward_correlation)
    best_code_paths.append(code_paths[best_sample_idx])

    logging.info(f"Max Success: {max_success}, Execute Rate: {execute_rate}, Max Success Reward Correlation: {max_success_reward_correlation}")
    logging.info(f"Best Generation ID: {best_sample_idx}")
    logging.info(f"GPT Output Content:\n" +  responses[best_sample_idx]["message"]["content"] + "\n")
    logging.info(f"User Content:\n" + best_content + "\n")

    np.savez('summary.npz', max_successes=max_successes, execute_rates=execute_rates, best_code_paths=best_code_paths, max_successes_reward_correlation=max_successes_reward_correlation)

    if len(messages) == 2:
        messages += [{"role": "assistant", "content": responses[best_sample_idx]["message"]["content"]}]
        messages += [{"role": "user", "content": best_content}]
    else:
        assert len(messages) == 4
        messages[-2] = {"role": "assistant", "content": responses[best_sample_idx]["message"]["content"]}
        messages[-1] = {"role": "user", "content": best_content}

    # Save dictionary as JSON file
    with open('messages.json', 'w') as file:
        json.dump(messages, file, indent=4)
    

if __name__ == "__main__":
    main()