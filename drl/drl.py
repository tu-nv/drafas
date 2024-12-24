import simpy
import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from concurrent.futures import ThreadPoolExecutor
import os, argparse
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.ppo_mask import MaskablePPO
import torch
from stable_baselines3 import PPO, A2C, DQN
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnNoModelImprovement

from simpy_env import Service, Client, SimpyDrafasEnv
from threshold_autoscaler import threshold_autoscaler_v1, threshold_autoscaler_v2
from env import DrafasEnv
from drl_utils import EvalStats, TOTAL_GPU, service_configs, CUSTOM_TEST_DIR

import itertools
import random, time
from datetime import datetime

from skopt import gp_minimize
import optuna
from skopt.space import Real, Categorical
from math import sqrt

random.seed(42)


BASE_DIR = os.path.dirname(os.path.realpath(__file__))

parser = argparse.ArgumentParser(description='simpy simulator')

parser.add_argument('--train_dir', type=str, default="trace/train", help='relative path to the train dir')
parser.add_argument('--val_dir', type=str, default="trace/val", help='relative path to the validation dir')
parser.add_argument('--test_dir', type=str, default="trace/test", help='relative path to the test dir')
parser.add_argument('--start_point', type=int, default=0, help='start from this timestamp point')
parser.add_argument('--model', type=str, default="MPPO", choices=["PPO", "MPPO", "DQN"], help='choice of model')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='learning rate')
parser.add_argument('--log_dir', type=str, default="./tensorboard_log", help='tensorboard log dir')
parser.add_argument('--log_name', type=str, default=None, help='custom tensorboard log name')
parser.add_argument('--best_model_log_name', type=str, default=None, help='custom best model log name')
parser.add_argument('--mode', type=str, default="train", choices=["train", "test", "real_env", "parameter_search"], help='running mode, train or test')
parser.add_argument('--alg', type=str, default="all", choices=["drl", "threshold", "static", "all"], help='algorithm to test')
parser.add_argument('--service', type=str, default="pytorch", choices=["ollama", "whisper", "triton", "coqui", "pytorch", "all"], help='service to train')

args = parser.parse_args()

# Initialize multiple services and environments

services = {}
envs = {}
val_services = {}
val_envs = {}
test_services = {}
test_envs = {}
real_envs = {}


def mask_fn(env: gym.Env) -> np.ndarray:
    return env.valid_action_mask()

# Create services, clients, and environments
for service_name, service_conf in service_configs.items():
    init_inst, gpu_per_inst, capacity_per_inst, proc_time_gpu_ms, delay_sla, startup_time, req_scaling, rate_limit_req_per_duration, rate_limit_duration = service_conf

    simpy_env = simpy.Environment()
    service = Service(simpy_env, init_inst, gpu_per_inst, capacity_per_inst, proc_time_gpu_ms, delay_sla, startup_time, rate_limit_req_per_duration, rate_limit_duration)
    client = Client(simpy_env, service, f'{BASE_DIR}/{args.train_dir}', req_scaling)
    env = SimpyDrafasEnv(service_name, simpy_env, service, client, services, 'train')
    envs[service_name] = ActionMasker(env, mask_fn)
    services[service_name] = service

    val_simpy_env = simpy.Environment()
    val_service = Service(val_simpy_env, init_inst, gpu_per_inst, capacity_per_inst, proc_time_gpu_ms, delay_sla, startup_time, rate_limit_req_per_duration, rate_limit_duration)
    val_client = Client(val_simpy_env, val_service, f'{BASE_DIR}/{args.val_dir}', req_scaling)
    val_env = SimpyDrafasEnv(service_name, val_simpy_env, val_service, val_client, val_services, 'val')
    val_envs[service_name] = ActionMasker(val_env, mask_fn)
    val_services[service_name] = val_service

    test_simpy_env = simpy.Environment()
    test_service = Service(test_simpy_env, init_inst, gpu_per_inst, capacity_per_inst, proc_time_gpu_ms, delay_sla, startup_time, rate_limit_req_per_duration, rate_limit_duration)
    test_client = Client(test_simpy_env, test_service, f'{BASE_DIR}/{args.test_dir}/{CUSTOM_TEST_DIR[service_name]}', req_scaling)
    # test_client = Client(test_simpy_env, test_service, f'{BASE_DIR}/{args.val_dir}', req_scaling)
    test_env = SimpyDrafasEnv(service_name, test_simpy_env, test_service, test_client, test_services, 'test')
    test_envs[service_name] = ActionMasker(test_env, mask_fn)
    test_envs[service_name] = test_env
    test_services[service_name] = test_service

    real_env = DrafasEnv(service_name, delay_sla)
    real_envs[service_name] = ActionMasker(real_env, mask_fn)

# Training function to be run in parallel
def train_model(service_name):
    print(f"Training service {service_name}")
    env = envs[service_name]
    val_env = val_envs[service_name]
    check_env(env, warn=True)

    policy_kwargs = dict(activation_fn=torch.nn.ReLU,
                     net_arch=dict(pi=[64, 64], vf=[64, 64]))

    def lr_schedule(progress_remaining):
        lr = 0.0002 * progress_remaining
        print(f"----------lr: {lr}")
        return lr

    if args.model == 'MPPO':
        model = MaskablePPO(MaskableActorCriticPolicy, env, policy_kwargs=policy_kwargs,
                            learning_rate=lr_schedule, tensorboard_log=args.log_dir, verbose=1)
    elif args.model == 'PPO':
        model = PPO("MlpPolicy", env, policy_kwargs=policy_kwargs,
                    learning_rate=args.learning_rate, tensorboard_log=args.log_dir, verbose=1)
    elif args.model == 'DQN':
        model = DQN("MlpPolicy", env,
                    learning_rate=args.learning_rate, tensorboard_log=args.log_dir, verbose=1)

    stop_train_callback = StopTrainingOnNoModelImprovement(max_no_improvement_evals=4, min_evals=8, verbose=1)
    best_model_log_path = f"{BASE_DIR}/best_model_logs/{env.name}" if args.best_model_log_name is None else f"{BASE_DIR}/best_model_logs/{args.best_model_log_name}"
    val_callback = MaskableEvalCallback(val_env,
                                 best_model_save_path=best_model_log_path,
                                log_path=best_model_log_path,
                                eval_freq=int(24*3600/30 * 5),
                                callback_after_eval=stop_train_callback,
                                deterministic=True,
                                n_eval_episodes=3,
                                render=False)

    print(model.policy)
    tb_log_name = f"{service_name}_{args.model}" if args.log_name is None else args.log_name
    model.learn(total_timesteps=300_000,
                callback=val_callback,
                tb_log_name=tb_log_name
                )
    return model


def test_model(service_name):
    num_test_steps = int(24*3600/30) # 1 day
    test_stats = EvalStats()

    if args.alg == 'all' or args.alg == 'drl':
        print("--------------DRL-------------")
        model = MaskablePPO.load(f"{BASE_DIR}/best_model_logs/{service_name}/best_model.zip")
        obs, _ = test_envs[service_name].reset()

        for step in range(num_test_steps):
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = test_envs[service_name].step(action)
            test_stats.add_observation(obs, info)
            if terminated or truncated:
                obs, _ = test_envs[service_name].reset()
            # print(f"{service_name}: All time at step: {step}: {test_stats.summary_str()}")
        print(f"{service_name}: {test_stats.summary_str()}")
        test_stats.write_stats(f"stats/simu-DRL-{service_name}.txt")
        test_stats.clear()

    if args.alg == 'all' or args.alg == 'threshold':
        print("--------------threshold autoscaler-------------")
        # threshold autoscaler test
        obs, _ = test_envs[service_name].reset()
        for step in range(num_test_steps):
            sla_violation_rate = obs[-2]
            inst_util = obs[-4]
            num_instance = obs[-1]
            action = threshold_autoscaler_v2(sla_violation_rate, inst_util, service=service_name)
            obs, reward, terminated, truncated, info = test_envs[service_name].step(action)
            test_stats.add_observation(obs, info)
            if terminated or truncated:
                obs, _ = test_envs[service_name].reset()
            # print(f"{service_name}: All time at step: {step}: {test_stats.summary_str()}")
        print(f"{service_name}: {test_stats.summary_str()}")
        test_stats.write_stats(f"stats/simu-TH-{service_name}.txt")
        test_stats.clear()

    if args.alg == 'all' or args.alg == 'static':
        print("--------------static (no auto scaling)-------------")
        obs, _ = test_envs[service_name].reset()
        for step in range(num_test_steps):
            sla_violation_rate = obs[-2]
            inst_util = obs[-4]
            num_instance = obs[-1]
            action = 2 # maintain
            obs, reward, terminated, truncated, info = test_envs[service_name].step(action)
            test_stats.add_observation(obs, info)
            if terminated or truncated:
                obs, _ = test_envs[service_name].reset()
            # print(f"{service_name}: All time at step: {step}: {test_stats.summary_str()}")
        print(f"{service_name}: {test_stats.summary_str()}")
        # test_stats.write_stats(f"stats/simu-static-{service_name}.txt")
        test_stats.clear()

def test_model_real_env(service_name):
    num_test_steps = int(24*3600/30) # 1 day
    test_stats = EvalStats()

    if args.alg == 'drl':
        print("--------------DRL-------------")
        model = MaskablePPO.load(f"{BASE_DIR}/best_model_logs/{service_name}/best_model.zip")
        obs, _ = real_envs[service_name].reset()

        for step in range(num_test_steps):
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = real_envs[service_name].step(action)
            test_stats.add_observation(obs, info)
            if terminated or truncated:
                obs, _ = real_envs[service_name].reset()
            # print(f"{service_name}: time {step * 30}: {test_stats.summary_str()}")
        print(f"{service_name}: {test_stats.summary_str()}")
        test_stats.write_stats(f"stats/real-env-DRL-{service_name}.txt")
        test_stats.clear()

    elif args.alg == 'threshold':
        print("--------------threshold autoscaler-------------")
        # threshold autoscaler test
        obs, _ = real_envs[service_name].reset()
        for step in range(num_test_steps):
            sla_violation_rate = obs[-2]
            inst_util = obs[-4]
            num_instance = obs[-1]
            print(f"first_sla: {sla_violation_rate}")
            action = threshold_autoscaler_v1(sla_violation_rate, inst_util, service=service_name)
            obs, reward, terminated, truncated, info = real_envs[service_name].step(action)
            test_stats.add_observation(obs, info)
            if terminated or truncated:
                obs, _ = real_envs[service_name].reset()
            # print(f"{service_name}: time {step * 30}: {test_stats.summary_str()}")
        print(f"{service_name}: {test_stats.summary_str()}")
        test_stats.write_stats(f"stats/real-env-TH-{service_name}.txt")
        test_stats.clear()
    else:
        print(f"In real env test, --alg need to be either drl or threshold. current one is {args.alg}")


if args.mode == "train":
    if args.service != "all":
        model = train_model(args.service)
    else:
        with ThreadPoolExecutor() as executor:
            futures = {executor.submit(train_model, service_name): service_name for service_name in envs.keys()}

elif args.mode == "test":
    if args.service != "all":
        test_model(args.service)
    else:
        with ThreadPoolExecutor() as executor:
            futures = {executor.submit(test_model, service_name): service_name for service_name in envs.keys()}

elif args.mode == "real_env":
    if args.service != "all":
        test_model_real_env(args.service)
    else:
        with ThreadPoolExecutor() as executor:
            futures = {executor.submit(test_model_real_env, service_name): service_name for service_name in real_envs.keys()}


elif args.mode == "parameter_search":
    print(f"start parameter search for: {args.service}")
    num_test_steps = int(24*3600/30) # 1 day
    test_stats = EvalStats()

    # Define the evaluation function
    def evaluate(trial):
        # Sample parameters from the search space
        sla_high = trial.suggest_float("sla_high", 0.005, 0.1, step=0.002)
        sla_low = trial.suggest_float("sla_low", 0.0001, 0.001, step=0.0001)
        gpu_high = trial.suggest_float("inst_util_high", 0.6, 0.9, step=0.02)
        gpu_low = trial.suggest_float("inst_util_low", 0.2, 0.5, step=0.02)

        # Initialize environment and stats
        obs, _ = val_envs[args.service].reset()
        test_stats.clear()
        # sla_violation_rate_norms = []

        for _ in range(num_test_steps):
            sla_violation_rate = obs[-2]
            inst_util = obs[-4]
            action = threshold_autoscaler_v2(sla_violation_rate, inst_util, None, sla_high, sla_low, gpu_high, gpu_low)
            obs, reward, terminated, truncated, info = val_envs[args.service].step(action)
            test_stats.add_observation(obs, info)

            if terminated or truncated:
                obs, _ = val_envs[args.service].reset()

        # Calculate performance metrics
        avg_sla_violation_rate, avg_num_instance, avg_proc_time, avg_inst_util = test_stats.summary()
        print(test_stats.summary_str())

        # Normalize the metrics
        avg_proc_time_norm = min(1, avg_proc_time / 10)
        num_inst_norm = avg_num_instance / TOTAL_GPU
        # avg_sla_violation_rate_norm = sum(sla_violation_rate_norms)/len(sla_violation_rate_norms)


        # Define the objective function (to minimize)
        reward_need_to_minimize = (
            0.9 * avg_sla_violation_rate +
            0.1 * num_inst_norm
        )

        return reward_need_to_minimize

    # def evaluate(trial):
    #     # Sample parameters from the search space
    #     inst_util_high_1 = trial.suggest_float("inst_util_high_1", 0.6, 0.8, step=0.01)
    #     inst_util_high_2 = trial.suggest_float("inst_util_high_2", inst_util_high_1, 0.95, step=0.01)
    #     inst_util_low_2 = trial.suggest_float("inst_util_low_2", 0.05, 0.3, step=0.01)
    #     inst_util_low_1 = trial.suggest_float("inst_util_low_1", inst_util_low_2, 0.5, step=0.01)

    #     # Initialize environment and stats
    #     obs, _ = val_envs[args.service].reset()
    #     test_stats.clear()
    #     # sla_violation_rate_norms = []

    #     for _ in range(num_test_steps):
    #         sla_violation_rate = obs[-2]
    #         inst_util = obs[-4]
    #         action = threshold_autoscaler(inst_util, None, inst_util_high_2, inst_util_high_1, inst_util_low_1, inst_util_low_2)
    #         obs, reward, terminated, truncated, info = val_envs[args.service].step(action)
    #         test_stats.add_observation(obs, info)

    #         if terminated or truncated:
    #             obs, _ = val_envs[args.service].reset()

    #     # Calculate performance metrics
    #     avg_sla_violation_rate, avg_num_instance, avg_proc_time, avg_inst_util = test_stats.summary()
    #     print(test_stats.summary_str())

    #     # Normalize the metrics
    #     avg_proc_time_norm = min(1, avg_proc_time / 10)
    #     num_inst_norm = avg_num_instance / TOTAL_GPU
    #     # avg_sla_violation_rate_norm = sum(sla_violation_rate_norms)/len(sla_violation_rate_norms)


    #     # Define the objective function (to minimize)
    #     reward_need_to_minimize = (
    #         0.9 * avg_sla_violation_rate +
    #         # 0.05 * avg_proc_time_norm +
    #         0.1 * num_inst_norm
    #     )

    #     return reward_need_to_minimize

    # Create an Optuna study
    study = optuna.create_study(direction="minimize")

    # Run the optimization
    study.optimize(evaluate, n_trials=100)

    # Print the best parameters and their corresponding value
    best_params = study.best_params
    best_value = study.best_value
    print(f"{args.service}: Best Parameters: {best_params}, Best Objective Value: {best_value}")



else:
    print(f"wrong run mode: {args.mode}")
    exit(1)
# Use ThreadPoolExecutor to train models in parallel
# with ThreadPoolExecutor() as executor:
#     futures = {executor.submit(train_model, env, i): i for i, env in enumerate(envs)}
    # for future in futures:
    #     model, env_id = future.result()
    #     print(f"Training completed for environment {env_id}")

    #     # Test the trained agent
    #     obs, _ = environments[env_id].reset()
    #     for _ in range(10):
    #         action, _states = model.predict(obs, deterministic=True)
    #         obs, reward, terminated, truncated, info = environments[env_id].step(action)
    #         environments[env_id].render()
    #         if terminated or truncated:
    #             obs, _ = environments[env_id].reset()
