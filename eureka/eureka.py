import hydra
import numpy as np 
import json
import logging 
import matplotlib.pyplot as plt
import os
import openai
import re
import subprocess
from pathlib import Path
import shutil
import time 
import torch
import threading
from utils.misc import * 
from utils.extract_task_code import *

EUREKA_ROOT_DIR = os.getcwd()
ROOT_DIR = f"{EUREKA_ROOT_DIR}/.."

semaphore = threading.Semaphore(2)

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

    env_name = cfg.env.env_name.lower()
    task_rew_file = f'{ROOT_DIR}/{env_name}/{cfg.env.reward_template_file}'
    task_obs_file = f'{EUREKA_ROOT_DIR}/envs/{env_name}.py'
    shutil.copy(task_obs_file, f"env_init_obs.py")
    task_rew_code_string = file_to_string(task_rew_file)
    task_obs_code_string = file_to_string(task_obs_file)
    output_file = f"{ROOT_DIR}/{env_name}/{cfg.env.reward_output_file}"

    # Loading all text prompts
    prompt_dir = f'{EUREKA_ROOT_DIR}/prompts'
    initial_system = file_to_string(f'{prompt_dir}/initial_system.txt')
    code_output_tip = file_to_string(f'{prompt_dir}/code_output_tip.txt')
    code_feedback = file_to_string(f'{prompt_dir}/code_feedback.txt')
    initial_user = file_to_string(f'{prompt_dir}/initial_user.txt')
    reward_signature = file_to_string(f'{prompt_dir}/reward_signatures/{env_name}.txt')
    policy_feedback = file_to_string(f'{prompt_dir}/policy_feedback.txt')
    execution_error_feedback = file_to_string(f'{prompt_dir}/execution_error_feedback.txt')

    initial_system = initial_system.format(task_reward_signature_string=reward_signature) + code_output_tip
    initial_user = initial_user.format(task_obs_code_string=task_obs_code_string, task_description=task_description)
    messages = [{"role": "system", "content": initial_system}, {"role": "user", "content": initial_user}]

    DUMMY_FAILURE = -10000.
    max_successes = []
    max_successes_reward_correlation = []
    execute_rates = []
    best_code_paths = []
    max_success_overall = DUMMY_FAILURE
    max_success_reward_correlation_overall = DUMMY_FAILURE
    max_reward_code_path = None 
    
    # Eureka generation loop
    for iter in range(cfg.iteration):
        # Get Eureka response
        responses = []
        response_cur = None
        total_samples = 0
        total_token = 0
        total_completion_token = 0
        chunk_size = cfg.sample if "gpt-3.5" in model else 4

        logging.info(f"Iteration {iter}: Generating {cfg.sample} samples with {cfg.model}")

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

        # # Loading pre-queried reward functions
        # reward_dir = os.path.join(EUREKA_ROOT_DIR,"saved_rewards")
        # responses = torch.load(os.path.join(reward_dir,"responses.pt"))
        # prompt_tokens = torch.load(os.path.join(reward_dir,"prompt_tokens.pt"))
        # total_completion_token = torch.load(os.path.join(reward_dir,"total_completion_token.pt"))
        # total_token = torch.load(os.path.join(reward_dir,"total_token.pt"))

        if cfg.sample == 1:
            logging.info(f"Iteration {iter}: GPT Output:\n " + responses[0]["message"]["content"] + "\n")

        # Logging Token Information
        logging.info(f"Iteration {iter}: Prompt Tokens: {prompt_tokens}, Completion Tokens: {total_completion_token}, Total Tokens: {total_token}")

        code_runs = [] 
        rl_runs = []
        for response_id in range(cfg.sample):
            response_cur = responses[response_id]["message"]["content"]
            logging.info(f"Iteration {iter}: Processing Code Run {response_id}")

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

            # Remove unnecessary imports
            lines = code_string.split("\n")
            lines = [" "*4 + line for line in lines]
            for i, line in enumerate(lines):
                if line.strip().startswith("def "):
                    code_string = "\n".join(lines[i:])
                    break
            
            code_runs.append(code_string)
                    
            # Add the Eureka Reward Signature to the environment code
            cur_task_rew_code_string = task_rew_code_string.replace("# INSERT EUREKA REWARD HERE", code_string)

            # Save the new environment code when the output contains valid code string!
            with open(output_file, 'w') as file:
                file.writelines(cur_task_rew_code_string + '\n')

            with open(f"env_iter{iter}_response{response_id}_rewardonly.py", 'w') as file:
                file.writelines(code_string + '\n')

            # Copy the generated environment code to hydra output directory for bookkeeping
            shutil.copy(output_file, f"env_iter{iter}_response{response_id}.py")

            # Find the freest GPU to run GPU-accelerated RL
            set_freest_gpu()
            
            # Execute the python file with flags
            rl_filepath = f"env_iter{iter}_response{response_id}.txt"
            with open(rl_filepath, 'w') as f:
                command = f"python -u {ROOT_DIR}/{env_name}/{cfg.env.train_script} --iterations {cfg.env.train_iterations} --dr-config off --reward-config eureka"
                command = command.split(" ")
                if not cfg.use_wandb:
                    command.append("--no-wandb")
                process = subprocess.run(command, stdout=f, stderr=f)
            block_until_training(rl_filepath, success_keyword=cfg.env.success_keyword, failure_keyword=cfg.env.failure_keyword,
                                 log_status=True, iter_num=iter, response_id=response_id)
            rl_runs.append(process)

        # Gather RL training results and construct reward reflection
        code_feedbacks = []
        contents = []
        successes = []
        reward_correlations = []
        code_paths = []
        
        exec_success = False 
        for response_id, (code_run, rl_run) in enumerate(zip(code_runs, rl_runs)):
            rl_run.communicate()
            rl_filepath = f"env_iter{iter}_response{response_id}.txt"
            code_paths.append(f"env_iter{iter}_response{response_id}.py")
            try:
                with open(rl_filepath, 'r') as f:
                    stdout_str = f.read() 
            except: 
                content = execution_error_feedback.format(traceback_msg="Code Run cannot be executed due to function signature error! Please re-write an entirely new reward function!")
                content += code_output_tip
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
                
                epochs_per_log = 10
                content += policy_feedback.format(epoch_freq=epochs_per_log*epoch_freq)
                
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
                code_feedbacks.append(code_feedback)
                content += code_feedback
            else:
                # Otherwise, provide execution traceback error feedback
                successes.append(DUMMY_FAILURE)
                reward_correlations.append(DUMMY_FAILURE)
                content += execution_error_feedback.format(traceback_msg=traceback_msg)

            content += code_output_tip
            contents.append(content) 

        # Repeat the iteration if all code generation failed
        if not exec_success and cfg.sample != 1:
            execute_rates.append(0.)
            max_successes.append(DUMMY_FAILURE)
            max_successes_reward_correlation.append(DUMMY_FAILURE)
            best_code_paths.append(None)
            logging.info("All code generation failed! Repeat this iteration from the current message checkpoint!")
            continue

        # Select the best code sample based on the success rate
        best_sample_idx = np.argmax(np.array(successes))
        best_content = contents[best_sample_idx]
            
        max_success = successes[best_sample_idx]
        max_success_reward_correlation = reward_correlations[best_sample_idx]
        execute_rate = np.sum(np.array(successes) >= 0.) / cfg.sample

        # Update the best Eureka Output
        if max_success > max_success_overall:
            max_success_overall = max_success
            max_success_reward_correlation_overall = max_success_reward_correlation
            max_reward_code_path = code_paths[best_sample_idx]

        execute_rates.append(execute_rate)
        max_successes.append(max_success)
        max_successes_reward_correlation.append(max_success_reward_correlation)
        best_code_paths.append(code_paths[best_sample_idx])

        logging.info(f"Iteration {iter}: Max Success: {max_success}, Execute Rate: {execute_rate}, Max Success Reward Correlation: {max_success_reward_correlation}")
        logging.info(f"Iteration {iter}: Best Generation ID: {best_sample_idx}")
        logging.info(f"Iteration {iter}: GPT Output Content:\n" +  responses[best_sample_idx]["message"]["content"] + "\n")
        logging.info(f"Iteration {iter}: User Content:\n" + best_content + "\n")
            
        # Plot the success rate
        fig, axs = plt.subplots(2, figsize=(6, 6))
        fig.suptitle(f'{task}')

        x_axis = np.arange(len(max_successes))

        axs[0].plot(x_axis, np.array(max_successes))
        axs[0].set_title("Max Success")
        axs[0].set_xlabel("Iteration")

        axs[1].plot(x_axis, np.array(execute_rates))
        axs[1].set_title("Execute Rate")
        axs[1].set_xlabel("Iteration")

        fig.tight_layout(pad=3.0)
        plt.savefig('summary.png')
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
    
    if max_reward_code_path is None: 
        logging.info("All iterations of code generation failed, aborting...")
        logging.info("Please double check the output env_iter*_response*.txt files for repeating errors!")
        exit()
    logging.info(f"Task: {task}, Max Training Success {max_success_overall}, Correlation {max_success_reward_correlation_overall}, Best Reward Code Path: {max_reward_code_path}")

    best_reward = file_to_string(max_reward_code_path)
    with open(output_file, 'w') as file:
        file.writelines(best_reward + '\n')
    
    # Get run directory of best-performing policy
    with open(max_reward_code_path.replace(".py", ".txt"), "r") as file:
        lines = file.readlines()
    for line in lines:
        if line.startswith("Dashboard: "):
            run_dir = line.split(": ")[1].strip()
            run_dir = run_dir.replace("http://app.dash.ml/", f"{ROOT_DIR}/{env_name}/runs/")
            logging.info("Best policy run directory: " + run_dir)

if __name__ == "__main__":
    main()