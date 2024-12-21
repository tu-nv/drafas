import simpy
import gymnasium as gym
import numpy as np
from collections import deque
import os
import random
from prometheus_api_client import PrometheusConnect
from datetime import datetime, timedelta
from kubernetes import client, config
import time
from drl_utils import TOTAL_GPU, USAGE_AVG_PERIOD_SEC, SCALE_DOWN_COOLDOWN_STEP, ENVOY_INBOUND, service_configs




# Custom SimPy-Gym Environment with Queue Management for each Service
class DrafasEnv(gym.Env):
    """
    Custom Gymnasium Environment that models system metrics over time.
    state: inst_util, continuous value between 0 and 1.
        is_resource_full, value can be 1 or 0.
        sla_violation_rate, continuous value, between 0 and 1.
        avg_proc_time, continuous value, between 0 and 1.
        number_inst, integer, between 1 and 20.
        the state contain the history size of 5, i.e., store last 5 value of each metric.
    action: -1, 0, or 1
    reward: design a reward that reduce sla_violation_rate, reduce the number_inst, and reduce the avg_proc_time
    """
    def __init__(self, service_name, proc_time_sla):
        super(DrafasEnv, self).__init__()

        prometheus_url = "http://141.223.124.62:30090"
        self.prom = PrometheusConnect(url=prometheus_url, disable_ssl=True)  # disable_ssl=True if not using HTTPS

        configuration = client.Configuration()
        configuration.host = "http://141.223.124.62:50192"
        self.kube_core_api = client.CoreV1Api(client.ApiClient(configuration))
        self.kube_apps_api = client.AppsV1Api(client.ApiClient(configuration))

        self.service_name = service_name
        self.proc_time_sla = proc_time_sla
        self.ns = "default"

        # Define observation space
        self.history_size = 5  # Number of historical steps
        # Each metric has its own range
        self.observation_space = gym.spaces.Box(
            low=np.array([0, 0, 0, 0, 0,] * self.history_size, dtype=np.float32),
            high=np.array([1, 1, 1, 1, 1,] * self.history_size, dtype=np.float32),
            dtype=np.float32
        )

        self.action_space = gym.spaces.Discrete(3)

        # Initialize state (history of metrics)
        self.state = np.zeros((self.history_size, 5), dtype=np.float32)

        # Simulation parameters
        self.max_steps = int(24*3600/USAGE_AVG_PERIOD_SEC)
        self.current_step = 0

        self.sla_violation_rate = 0.0
        self.inst_util = 0.0
        self.num_instances = 1
        self.remain_gpus = TOTAL_GPU - self.num_instances

        self.num_instances_history = []
        self.sla_violation_rate_history = []
        self.inst_util_history = []

        self.time_after_current_sleep = time.monotonic()
        self.time_before_next_sleep = time.monotonic()

        self.scale_down_cooldown = SCALE_DOWN_COOLDOWN_STEP

    # Scale the deployment
    def scale_deployment(self, action):
        # Get the current deployment object
        deployment_name = f"{self.service_name}-deployment"
        deployment = self.kube_apps_api.read_namespaced_deployment(name=deployment_name, namespace=self.ns)

        # Update the replicas count
        if action > 0:
            deployment.spec.replicas += action
        elif action < 0:
            if deployment.spec.replicas + action < 1:
                print(f"{self.service_name}: cannot scale down, action {action}, replicas {deployment.spec.replicas} ", flush=True)
                return False
            deployment.spec.replicas += action
        elif action == 0:
            # maintain
            return True
        else:
            print(f"{self.service_name}: invalid action {action}")
            return False

        # Apply the changes to the deployment
        api_response = self.kube_apps_api.patch_namespaced_deployment(name=deployment_name, namespace=self.ns, body=deployment)
        print(f"Deployment '{deployment_name}' scaled to {deployment.spec.replicas} replicas after action {action}", flush=True)
        return True


    def exec_query(self, query_str):
        try:
            result = self.prom.custom_query(query=query_str)
            if result:
                for entry in result:
                    value = entry["value"]
                    timestamp = float(value[0])
                    # print(f"{query_str}\nTimestamp: {timestamp}: {value[1]}")
                    if value[1] == "NaN":
                        return 0
                    else:
                        return float(value[1])
            else:
                print(f"No data found for query {query_str}")
                return 0
        except Exception as e:
            print("Error querying Prometheus:", e)

        return None

    def get_stats(self):
        # Initialize GPU usage counter
        gpu_used = 0
        total_gpu = 0
        num_instances = 0

        pods = self.kube_core_api.list_namespaced_pod(self.ns)
        for pod in pods.items:
            for container in pod.spec.containers:
                if container.name == self.service_name and pod.status.phase == 'Running':
                    num_instances += 1
                # Check if the container requests GPU resources
                gpu_request = container.resources.requests.get("nvidia.com/gpu") if container.resources.requests else None
                if gpu_request:
                    gpu_used += int(gpu_request)

        # nodes = self.kube_core_api.list_node()
        # for node in nodes.items:
        #     total_gpu += int(node.status.allocatable.get("nvidia.com/gpu"))

        remain_gpus = TOTAL_GPU - gpu_used

        # Define the metric name and label filter
        # gpu_util_query = f'avg_over_time(DCGM_FI_DEV_GPU_UTIL{{container="{self.service_name}"}}[30s])'
        avg_proc_time_query = f'sum(increase(istio_request_duration_milliseconds_sum{{container="{self.service_name}", response_code="200"}}[30s])) / sum(increase(istio_request_duration_milliseconds_count{{container="{self.service_name}", response_code="200"}}[30s])) / 1000'
        # istio report label for 'le' ,or proc_time_sla bucket, has one 0 digit after dot, like 500.0, not 500
        sla_violation_query = f'1 - sum(increase(istio_request_duration_milliseconds_bucket{{container="{self.service_name}", le="{self.proc_time_sla}.0", response_code="200"}}[30s])) / sum(increase(istio_request_duration_milliseconds_bucket{{container="{self.service_name}", le="+Inf"}}[30s]))'

        req_per_s = f'sum(increase(istio_requests_total{{container="{self.service_name}"}}[30s]))/30'
        cap_per_inst = service_configs[self.service_name][2]
        service_cap = cap_per_inst * num_instances
        # only in-processing request should be counted.
        # need to use subquery syntax (30s:1s)
        # https://stackoverflow.com/questions/56242426/query-for-aggregation-of-uptime-of-multiple-apps-with-promql
        inst_util_query = f'sum(avg_over_time(clamp_max(envoy_http_downstream_rq_active{{container="{self.service_name}", http_conn_manager_prefix="{ENVOY_INBOUND[self.service_name]}"}}, {cap_per_inst})[30s:1s])) / {num_instances}'

        # avg_inst_util = self.exec_query(gpu_util_query) / 100
        avg_proc_time = self.exec_query(avg_proc_time_query)
        sla_violation_rate = self.exec_query(sla_violation_query)
        req_per_s = self.exec_query(req_per_s)
        avg_inst_util = self.exec_query(inst_util_query)

        # Query the metric average over the last 30 seconds
        pods = self.kube_core_api.list_namespaced_pod(self.ns)




        return req_per_s, avg_inst_util, avg_proc_time, sla_violation_rate, num_instances, remain_gpus

    def step(self, action):
        # Map action from 0,1,2 to -1,0,1
        # print("--------------------")

        action = int(action) - 2
        invalid_action = 0
        action_mask = self.valid_action_mask()

        # if cannot scale up/down by 2 but can scale up/down by 1, do it
        if action == -2 and action_mask[0] == 0:
            if action_mask[1] == 1:
                action = -1
            else:
                invalid_action = 1
        elif action == -1 and action_mask[1] == 0:
            invalid_action = 1
        elif action == 2 and action_mask[4] == 0:
            if action_mask[3] == 1:
                action = 1
            else:
                invalid_action = 1
        elif action == 1 and action_mask[3] == 0:
            invalid_action = 1

        print(f"\n{self.service_name}: Step: {self.current_step}, action {action}, scale_cooldown: {self.scale_down_cooldown}, action mask {action_mask}, state:\n"
              f"req_per_inst, inst_util, action_mask, sla_violation_rate, num_inst_norm\n"
              f"{self.state}\n")

        if not invalid_action:
            self.scale_deployment(action)
            if action > 0:
                self.scale_down_cooldown = SCALE_DOWN_COOLDOWN_STEP
        else:
            print(f"{self.service_name}: invalid_action: {action}", flush=True)

        # count the time for all codes during step
        self.time_before_next_sleep = time.monotonic()
        execution_time = self.time_before_next_sleep - self.time_after_current_sleep
        time.sleep(USAGE_AVG_PERIOD_SEC - execution_time)
        self.time_after_current_sleep = time.monotonic()

        self.scale_down_cooldown -= 1

        req_per_s, avg_inst_util, avg_proc_time, sla_violation_rate, num_instances, remain_gpus = self.get_stats()
        avg_proc_time_norm = min(1, avg_proc_time / 10)
        num_inst_norm = num_instances/TOTAL_GPU
        static_proc_time_ms = service_configs[self.service_name][3]
        # req_per_s_norm = min(1, (req_per_s * static_proc_time_ms / 1000) / 2)
        req_per_s_norm = min(1, req_per_s/50)
        self.num_instances = num_instances
        self.remain_gpus = remain_gpus

        action_mask = self.valid_action_mask()
        action_mask = action_mask[0]<<4 | action_mask[1]<<3 | action_mask[2]<<2 | action_mask[3]<<1 | action_mask[4]
        action_mask = action_mask / 0b11111

        new_state = np.array([
                req_per_s_norm,
                # 0.5,
                avg_inst_util,
                # self.is_resource_full,
                action_mask,
                sla_violation_rate,
                num_inst_norm,
            ], dtype=np.float32)

        # Update state history
        self.state = np.roll(self.state, -1, axis=0)
        self.state[-1] = new_state

        # reward = - (0.9*sla_violation_rate + 0.05*avg_proc_time_norm + 0.05*num_inst_norm + 0.1*invalid_action)
        reward = - (sla_violation_rate + 0.05*num_inst_norm)

        # Check if the episode is done
        self.current_step += 1
        done = self.current_step >= self.max_steps

        self.num_instances_history.append(num_instances)
        self.inst_util_history.append(avg_inst_util)
        self.sla_violation_rate_history.append(sla_violation_rate)

        # if(self.current_step % 1 == 0):
        print(
            f"{self.service_name}: Step: {self.current_step}, prev_action: {action}, req_per_s_norm: {req_per_s_norm:.4f} avg_proc_time: {avg_proc_time:.5f}, "
              f"avg_inst_util: {avg_inst_util:.5f}, sla_violation_rate: {sla_violation_rate:.5f}, "
              f"num_inst: {self.num_instances}, action_mask: {self.valid_action_mask()}, execution_time: {execution_time}\n"
              f"All time: avg num_inst: {sum(self.num_instances_history)/len(self.num_instances_history):.5f}, "
              f"avg inst_util: {sum(self.inst_util_history)/len(self.inst_util_history):.5f}, "
              f"avg sla_violation: {sum(self.sla_violation_rate_history)/len(self.sla_violation_rate_history):.5f}\n"
              f"sla_violation_rate_history: {[round(x, 5) for x in self.sla_violation_rate_history]}\n",
            flush=True)
        # if(self.current_step % 1 == 0):
        #     print(
        #         f'{self.service_name}: All time: avg num_inst: {sum(self.num_instances_history) / (len(self.num_instances_history) + 0.00001):4f}, '
        #         f'avg sla_violation_rate: {sum(self.sla_violation_rate_history) / (len(self.sla_violation_rate_history) + 0.00001):4f}'
        #     )

        observation = self.state.flatten()
        info = {
            "avg_proc_time": avg_proc_time
        }

        return observation, reward, done, False, info

    def valid_action_mask(self):
        action_masks = np.ones((5,), dtype=int)
        if (self.num_instances == 2):
            action_masks[0] = 0
        if (self.num_instances == 1):
            action_masks[0] = 0
            action_masks[1] = 0

        if self.scale_down_cooldown > 0:
            action_masks[0] = 0
            action_masks[1] = 0

        if self.remain_gpus < 2:
            action_masks[4] = 0
        if self.remain_gpus < 1:
            action_masks[3] = 0
            action_masks[4] = 0

        return action_masks

    def reset(self, seed=None, options=None, full_reset = False):
        super().reset(seed=42)
        self.current_step = 0
        # Initialize state with observations
        for i in range(self.history_size):
            self.state[i] = np.array([
                0,
                0,
                0,
                0,
                1/TOTAL_GPU,
            ], dtype=np.float32)

        self.sla_violation_rate = 0.0
        self.inst_util = 0.0
        self.num_instances = 1

        self.num_instances_history = []
        self.sla_violation_rate_history = []
        self.inst_util_history = []

        observation = self.state.flatten()
        return observation, {}

    def render(self, mode='human'):
        # print(f"Queue length: {self.queue_length}, Service instances: {self.service_name.num_inst}")
        return

if __name__ == "__main__":
    test_env = DrafasEnv("triton", 400)
    test_env.get_stats()
    # test_env.scale_deployment(2)
    test_env.step(2)
    test_env.step(2)
