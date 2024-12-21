import gymnasium as gym
from gymnasium import spaces
import numpy as np
import time
import requests


class EmuDrafasEnv(gym.Env):
    """
    Custom Gymnasium Environment that models system metrics over time.
    state: gpu_usage, continuous value between 0 and 1.
        is_resource_full, value can be 1 or 0.
        sla_violation_rate, continuous value, between 0 and 1.
        avg_proc_time, continuous value, between 0 and 1.
        number_instances, integer, between 1 and 20.
        the state contain the history size of 5, i.e., store last 5 value of each metric.
    action: -1, 0, or 1
    reward: design a reward that reduce sla_violation_rate, reduce the number_instances, and reduce the avg_proc_time
    """
    metadata = {'render.modes': ['human']}


    def __init__(self, action_period, server_addr, client_report_addr, speedup_factor=1):
        super(EmuDrafasEnv, self).__init__()
        # Define action space: -1, 0, 1 mapped to 0, 1, 2
        self.action_space = spaces.Discrete(3)

        self.action_period = action_period
        self.speedup_factor = speedup_factor
        self.server_addr = server_addr
        self.client_report_addr = client_report_addr

        # Define observation space
        self.history_size = 5  # Number of historical steps
        # Each metric has its own range
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0, 1] * self.history_size, dtype=np.float32),
            high=np.array([1, 1, 1, 1, 20] * self.history_size, dtype=np.float32),
            dtype=np.float32
        )

        # Initialize state (history of metrics)
        self.state = np.zeros((self.history_size, 5), dtype=np.float32)

        # Simulation parameters
        self.max_steps = int(24*3600/15) # one day
        self.current_step = 0

        self.is_resource_full = 0
        self.num_instances = 1
        self.sla_violation_rate = 0.0
        self.gpu_usage = 0.0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        # Initialize state with observations
        for i in range(self.history_size):
            self.state[i] = np.array([
                0,
                0,
                0,
                0.1,
                1
            ], dtype=np.float32)

        observation = self.state.flatten()
        return observation, {}

    def step(self, action):
        # do action
        # Map action from 0,1,2 to -1,0,1
        action = int(action) - 1
        action_json = {'change': action}
        # action_json = {'change': 0}
        action_url = f"{self.server_addr}/scale"
        action_result = requests.post(action_url, json=action_json)

        # sleep before getting observation result for that action
        time.sleep(self.action_period / self.speedup_factor)

        # Obtain new observation from the environment
        service_report = requests.get(f"{self.server_addr}/stats")
        service_report = service_report.json()
        client_report = requests.get(f"{self.client_report_addr}/stats")
        client_report = client_report.json()

        gpu_usage = float(service_report["gpu_usage"])
        is_resource_full = 0 if int(service_report["remaining_gpus"]) >= 1 else 1
        sla_violation_rate = float(client_report["sla_violation_rate"])
        avg_proc_time = float(client_report["avg_processing_time_s"])
        avg_proc_time_norm = min(1, avg_proc_time / 10)
        number_instances = int(service_report["num_instances"])

        if self.current_step % 100 == 0:
            print(f"step: {self.current_step}, action: {action}, gpu: {gpu_usage:4f}, avg_proc_time: {avg_proc_time:4f}, sla_violation_rate: {sla_violation_rate:4f}, number_instances: {number_instances}")

        new_state = np.array([
                gpu_usage,
                is_resource_full,
                sla_violation_rate,
                avg_proc_time_norm,
                number_instances
            ], dtype=np.float32)

        # Update state history
        self.state = np.roll(self.state, -1, axis=0)
        self.state[-1] = new_state

        invalid_action = 0
        if (number_instances < 2 and action == -1) or (is_resource_full and action == 1):
            invalid_action = 1

        reward = - (0.4*sla_violation_rate + 0.1*avg_proc_time_norm + 0.2*(number_instances/20) + 0.3*invalid_action)

        # Check if the episode is done
        self.current_step += 1
        done = self.current_step >= self.max_steps
        # done = False

        # Return step information
        observation = self.state.flatten()
        info = {}
        truncated = False

        self.is_resource_full = is_resource_full
        self.num_instances = number_instances
        self.sla_violation_rate = sla_violation_rate
        self.gpu_usage = gpu_usage

        return observation, reward, done, truncated, info

    def valid_action_mask(self):
        action_masks = np.ones((3,), dtype=int)
        if self.num_instances < 2:
            action_masks[0] = 0
        if self.is_resource_full:
            action_masks[2] = 0
        return action_masks

    def render(self, mode='human'):
        # Optional: implement visualization
        pass

    def close(self):
        # Optional: implement any cleanup
        pass

if __name__ == "__main__":
    env = EmuDrafasEnv(action_period=15, server_addr="http://localhost:8101", client_report_addr="http://localhost:8201")
    env.step(2)
