import simpy
import gymnasium as gym
import numpy as np
from collections import deque
import os, time
import random
from math import sqrt
from drl_utils import SUCCESS, ERROR, TOTAL_GPU, USAGE_AVG_PERIOD_SEC, USAGE_AVG_PERIOD_MS, INST_UTIL_LOG_UPDATE_PERIOD, SCALE_DOWN_COOLDOWN_STEP

K3S_SERVICE_DISCOVERY_PERIOD_SEC = 30

# Service class with dynamic instances, processing time, and GPU usage tracking
class Service:
    def __init__(self, env: simpy.Environment, initial_inst, gpu_per_inst, capacity_per_inst, static_proc_time_range, req_delay_sla, startup_time, rate_limit_req_per_duration, rate_limit_duration):

        self.req_delay_sla = req_delay_sla
        self.static_proc_time_range = static_proc_time_range
        self.env = env
        self.initial_inst = initial_inst
        self.gpu_per_inst = gpu_per_inst
        self.capacity_per_inst = capacity_per_inst
        self.startup_time_s = int(startup_time / 1000)

        self.rate_limit_req_per_duration = rate_limit_req_per_duration
        self.rate_limit_duration = rate_limit_duration

        self.num_inst = self.initial_inst
        self.servers = [simpy.Resource(env, capacity=self.capacity_per_inst) for _ in range(self.initial_inst)]
        self.server_rate_limits = [self.rate_limit_req_per_duration for _ in range(self.initial_inst)]

        # Track request times and GPU usage
        self.sum_request_time = 0.0
        self.cnt_request_time = 0.0
        self.cnt_failed_req = 0
        self.cnt_sla_violated_req = 0
        # self.inst_util_log = deque(maxlen=INST_UTIL_LOG_LEN)  # Store recent GPU usage logs
        self.sum_inst_util = 0.0
        self.cnt_inst_util = 0
        self.round_robin_idx = 0
        self.new_instance_startup_cooldown = 0

        self.env_step = 0

        self.process = self.env.process(self.update_inst_util_log())
        self.env.process(self.new_instance_startup_cooldown_process())
        self.env.process(self.rate_limit_token_update_process())

        # self.req_cnt = 0
    def reset(self):
        self.servers = [simpy.Resource( self.env, capacity=self.capacity_per_inst) for _ in range(self.initial_inst)]
        # Track request times and GPU usage
        self.num_inst = self.initial_inst
        self.sum_request_time = 0.0
        self.cnt_request_time = 0.0
        self.cnt_failed_req = 0
        self.cnt_sla_violated_req = 0
        self.sum_inst_util = 0.0
        self.cnt_inst_util = 0
        self.round_robin_idx = 0
        # self.process.succeed()
        # self.process = self.env.process(self.update_inst_util_log())

    def rate_limit_token_update_process(self):
        while True:
            yield self.env.timeout(self.rate_limit_duration)
            num_server = len(self.servers)
            self.server_rate_limits = [self.rate_limit_req_per_duration for _ in range(num_server)]

    def new_instance_startup_cooldown_process(self):
        while True:
            yield self.env.timeout(1000)
            if self.new_instance_startup_cooldown > 0:
                self.new_instance_startup_cooldown -= 1

    def update_inst_util_log(self):
        while True:
            # sample every INST_UTIL_LOG_UPDATE_PERIOD
            yield self.env.timeout(INST_UTIL_LOG_UPDATE_PERIOD)
            total_req_in_proc = sum([x.count for x in self.servers])
            current_inst_util = total_req_in_proc / (self.num_inst * self.capacity_per_inst)
            # self.inst_util_log.append(current_inst_util)
            self.sum_inst_util += current_inst_util
            self.cnt_inst_util += 1


    def scale(self, action):
        if action == 1:
            self.servers.append(simpy.Resource(self.env, capacity=self.capacity_per_inst))
            self.server_rate_limits.append(self.rate_limit_req_per_duration)
            self.num_inst += 1
            self.new_instance_startup_cooldown = random.randint(self.startup_time_s, K3S_SERVICE_DISCOVERY_PERIOD_SEC)
            return SUCCESS
        elif action == 2:
            self.servers.append(simpy.Resource(self.env, capacity=self.capacity_per_inst))
            self.servers.append(simpy.Resource(self.env, capacity=self.capacity_per_inst))
            self.server_rate_limits.append(self.rate_limit_req_per_duration)
            self.server_rate_limits.append(self.rate_limit_req_per_duration)
            self.num_inst += 2
            self.new_instance_startup_cooldown = random.randint(self.startup_time_s, K3S_SERVICE_DISCOVERY_PERIOD_SEC) * 2
            return SUCCESS
        elif action == -1:
            if self.num_inst == 1:
                print("invalid action -1. cannot scale down", flush=True)
                return ERROR
            self.servers.pop()
            self.server_rate_limits.pop()
            self.num_inst -= 1
        elif action == -2:
            if self.num_inst == 2 or self.num_inst == 1:
                print("invalid action -2. cannot scale down", flush=True)
                return ERROR
            self.servers.pop()
            self.servers.pop()
            self.server_rate_limits.pop()
            self.server_rate_limits.pop()
            self.num_inst -= 2
        return SUCCESS

    def process_request(self):
        # yield self.env.process(self.service.process_request())
        start_time = self.env.now  # Record the request arrival time
        # Log GPU usage at the start of processing
        self.round_robin_idx += 1
        # do not allocate to the latest instance if it has not fully start yet
        if self.round_robin_idx >= self.num_inst or (self.new_instance_startup_cooldown > 0 and self.round_robin_idx == self.num_inst - 1):
            self.round_robin_idx = 0
        if len(self.servers[self.round_robin_idx].queue) > 10:
            # self.failed_reqs.append(start_time)
            self.cnt_failed_req += 1
            # self.req_cnt += 1
            return
        if self.server_rate_limits[self.round_robin_idx] <= 0:
            self.cnt_failed_req += 1
            return

        self.server_rate_limits[self.round_robin_idx] -= 1
        with self.servers[self.round_robin_idx].request() as request:
            yield request
            yield self.env.timeout(random.randint(*self.static_proc_time_range))
            # yield self.env.timeout(self.static_proc_time)
            end_time = self.env.now
            total_time = end_time - start_time
            # self.request_times.append((start_time, total_time))
            self.sum_request_time += total_time
            self.cnt_request_time += 1
            if total_time > self.req_delay_sla:
                self.cnt_sla_violated_req += 1
            # self.req_cnt += 1



    def get_service_stats(self):
        """Calculate average processing time and GPU usage over the last 30s."""
        # Calculate average processing time
        # now = self.env.now
        # somehow counting duration like below make total request less than actual.
        # therefore we just reset failed_reqs and request_times every call and assume that
        # it is called every USAGE_AVG_PERIOD_MS
        # recent_failed_reqs = [x for x in self.failed_reqs if now - x < USAGE_AVG_PERIOD_MS]
        # self.failed_reqs = recent_failed_reqs
        # recent_request_times = [x for x in self.request_times if now - x[0] < USAGE_AVG_PERIOD_MS]
        # self.request_times = recent_request_times
        num_total_req = self.cnt_failed_req + self.cnt_request_time
        avg_proc_time = 0.0
        if self.cnt_request_time > 0:
            avg_proc_time = self.sum_request_time / self.cnt_request_time
            avg_proc_time = avg_proc_time / 1000 # convert to seconds
            sla_violation_rate = (self.cnt_sla_violated_req + self.cnt_failed_req) / num_total_req
        elif num_total_req > 0:
            sla_violation_rate = self.cnt_failed_req / num_total_req
        else:
            sla_violation_rate = 0.0

        # Calculate average GPU usage over the last 30 time units
        # cutoff_time = now - USAGE_AVG_PERIOD_MS
        # relevant_usage = [usage for usage in self.inst_util_log if time >= cutoff_time]
        # avg_inst_util = sum(self.inst_util_log) / len(self.inst_util_log) if len(self.inst_util_log) > 0 else 0.0
        avg_inst_util = self.sum_inst_util / self.cnt_inst_util
        # avg_inst_util = avg_inst_util * random.randint(90, 110) / 100.0
        # avg_inst_util = max(0.0, min(avg_inst_util, 100.0))


        # print(self.env.now, int(self.req_cnt/30), int(num_total_req/30))
        # self.req_cnt = 0
        self.sum_request_time = 0.0
        self.cnt_request_time = 0
        self.cnt_failed_req = 0
        self.cnt_sla_violated_req = 0
        self.sum_inst_util = 0.0
        self.cnt_inst_util = 0

        return avg_proc_time, sla_violation_rate, avg_inst_util, num_total_req, self.cnt_failed_req


# Client class
class Client:
    def __init__(self, env: simpy.Environment, service: Service, trace_dir, req_scaling = 1):
        self.req_scaling = req_scaling
        self.env = env
        self.service = service
        self.trace_dir = trace_dir
        self.total_req = 0
        self.process = self.env.process(self.send_requests())  # Start the client process immediately

    def reset(self):
        self.total_req = 0
        self.process.interrupt()
        self.process = self.env.process(self.send_requests())

    def load_plan(self, file_path):
        plan = []
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    delay_str, num_requests_str = line.split()
                    delay = int(delay_str)
                    num_requests = int(num_requests_str)
                    plan.append((delay, num_requests))
        return plan

    def send_requests(self):
        try:
            while True:
                # for f in os.listdir(f'{BASE_DIR}/{args.plan_dir}'):
                # for plan_file in os.listdir(self.trace_dir):
                plan_file = random.choice(os.listdir(self.trace_dir))
                plan_file = os.path.join(self.trace_dir, plan_file)
                plan = self.load_plan(plan_file)
                print(f"Start with plan file {plan_file}", flush=True)
                for delay, num_requests in plan:
                    # no need to sleep delay because self.env.timeout(int(1000/num_requests)) already do it
                    # yield self.env.timeout(int(delay * 1000))
                    num_requests = int(num_requests * self.req_scaling)
                    self.total_req += num_requests
                    delay_bw_req = int(1000/num_requests) if num_requests > 0 else 0
                    # as delay_bw_rq is lower than 1000/num_request, there maybe some remaining delay
                    for _ in range(num_requests):
                        # spread requests evenly during next 1s
                        yield self.env.timeout(delay_bw_req)
                        self.env.process(self.service.process_request())
                    remain_delay_before_next_batch = delay * 1000 - delay_bw_req * num_requests
                    yield self.env.timeout(remain_delay_before_next_batch)
                    # print(remain_delay_before_next_batch)
        except simpy.Interrupt:
            print(f"client req sending process is reset")

#-----------------------------------------------------------------------------------------
# Custom SimPy-Gym Environment with Queue Management for each Service
class SimpyDrafasEnv(gym.Env):
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
    def __init__(self, name, simpy_env: simpy.Environment, service: Service, client: Client, all_services: dict[Service], mode):
        super(SimpyDrafasEnv, self).__init__()
        self.name = name
        self.env = simpy_env
        self.service = service
        self.client = client
        self.end_envt = self.env.event()
        self.all_services = all_services  # List of all services to manage total capacity
        self.mode = mode

        # Define observation space
        self.history_size = 5  # Number of historical steps
        self.state = np.zeros((self.history_size, 5), dtype=np.float32)
        # Each metric has its own range
        self.observation_space = gym.spaces.Box(
            low=np.array([ 0, 0, 0, 0, 0] * self.history_size, dtype=np.float32),
            high=np.array([ 1, 1, 1, 1, 1] * self.history_size, dtype=np.float32),
            dtype=np.float32
        )

        self.action_space = gym.spaces.Discrete(5)

        # Initialize state (history of metrics)

        # Simulation parameters
        self.max_steps = int(24*3600/(USAGE_AVG_PERIOD_MS/1000))
        self.current_step = 0

        self.is_resource_full = 0
        self.sla_violation_rate = 0.0
        self.inst_util = 0.0
        self.start_time = self.env.now

        self.scale_down_cooldown = SCALE_DOWN_COOLDOWN_STEP
        self.should_wait_other_services = True

    def get_remain_gpus(self):
        total_used_gpu = 0
        for service in self.all_services.values():
            total_used_gpu += service.num_inst * service.gpu_per_inst
        remain_gpu = TOTAL_GPU - total_used_gpu
        if self.mode == 'train':
            if random.choice([True, False]):
                # print(f"remain_gpu {remain_gpu}, total_gpu {total_used_gpu}")
                remain_gpu = random.randint(0, remain_gpu)

        return remain_gpu

        # if self.mode == 'train':
        #     threshold = random.randint(1, TOTAL_GPU)
        #     # threshold = TOTAL_GPU if random.choice([True, False]) else threshold
        # else:
        #     threshold = TOTAL_GPU

        # # print(f"total gpu: {self.mode} {total_used_gpu}, is resource full: {total_used_gpu < threshold}")
        # if total_used_gpu < threshold:
        #     return 0
        # else:
        #     return 1

    def check_if_ahead_other_services(self):
        is_ahead = False
        for service in self.all_services.values():
            if self.service.env_step > service.env_step:
                is_ahead = True
                break
        return is_ahead


    def step(self, action):
        self.service.env_step += 1
        # waif for other service to sync step. If wait for more than 2 secs, it means that the other service stop, so do not wait further
        start_wait_time = time.monotonic()
        while self.should_wait_other_services and self.check_if_ahead_other_services():
            time.sleep(0.001)
            if (time.monotonic() - start_wait_time > 1):
                self.should_wait_other_services = False

        # print(f"\n{self.name}: Step: {self.current_step}, action {int(action) - 1}, scale_cooldown: {self.scale_down_cooldown} state:\n"
        #       f"req_per_inst, inst_util, is_resource_full, sla_violation_rate, 0.5, num_inst_norm\n"
        #       f"{self.state}\n")

        # Map action from 0,1,2 to -1,0,1
        invalid_action = 0
        action = int(action) - 2
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

        if not invalid_action:
            # print(f"action {action}, invalid_action {invalid_action}, action mask:\n{action_mask}\n")
            self.service.scale(action)
            if action > 0:
                self.scale_down_cooldown = SCALE_DOWN_COOLDOWN_STEP
        # else:
        #     print(f"invalid action: {action}", flush=True)


        # Step the SimPy environment
        self.env.run(until=self.env.now + USAGE_AVG_PERIOD_MS)

        self.scale_down_cooldown -= 1

        # Calculate average processing time and GPU usage
        avg_proc_time, sla_violation_rate, avg_inst_util, num_recent_total_req, num_recent_failed_req = self.service.get_service_stats()
        avg_proc_time_norm = min(1, avg_proc_time / 10)
        self.is_resource_full = self.get_remain_gpus()
        num_inst_norm = self.service.num_inst/TOTAL_GPU
        # sla_violation_rate = sqrt(sla_violation_rate)
        # sla_violation_rate = sla_violation_rate
        # TODO:
        # avg_req_per_s = min(1, (num_recent_total_req * self.service.static_proc_time / 1000) / (self.service.num_inst * USAGE_AVG_PERIOD_SEC * 2))
        avg_req_per_s = min(1, num_recent_total_req / (USAGE_AVG_PERIOD_SEC * 50))
        # num_recent_total_req_norm = min(1, num_recent_total_req / 200 / 30)
        # num_recent_failed_req_norm = min(1, num_recent_failed_req / 20 / 30)

        action_mask = self.valid_action_mask()
        action_mask = action_mask[0]<<4 | action_mask[1]<<3 | action_mask[2]<<2 | action_mask[3]<<1 | action_mask[4]
        action_mask = action_mask / 0b11111

        new_state = np.array([
                avg_req_per_s,
                # 0.5,
                # avg_proc_time_norm,
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
        # reward = - (0.9*sla_violation_rate + 0.05*num_inst_norm + 0.05*invalid_action)
        reward = - (0.9*sla_violation_rate + 0.1*num_inst_norm)

        # Check if the episode is done
        self.current_step += 1
        done = self.current_step >= self.max_steps
        # done = False
        # Log every 100 step
        # if(self.current_step % 1 == 0):
        #     print(
        #         f"{self.name}: intermediate: Timestamp: {int((self.env.now - self.start_time)/1000)}, step: {self.current_step}, "
        #         f"prev_action: {action}, avg_req: {avg_req_per_s:.5f} avg_proc_time: {avg_proc_time:.5f}, "
        #         f"avg_inst_util: {avg_inst_util:.5f}, sla_violation_rate: {sla_violation_rate:.5f}, num_inst: {self.service.num_inst}, "
        #         f"action_mask: {self.valid_action_mask()}, rw: {reward:.5f}",
        #         flush=True)

        observation = self.state.flatten()
        info = {
            "avg_proc_time": avg_proc_time
        }

        return observation, reward, done, False, info

    def valid_action_mask(self):
        action_masks = np.ones((5,), dtype=int)
        remain_gpus = self.get_remain_gpus()

        if (self.service.num_inst == 2):
            action_masks[0] = 0
        if (self.service.num_inst == 1):
            action_masks[0] = 0
            action_masks[1] = 0

        if self.scale_down_cooldown > 0:
            action_masks[0] = 0
            action_masks[1] = 0

        if remain_gpus == 1:
            action_masks[4] = 0
        if remain_gpus == 0:
            action_masks[3] = 0
            action_masks[4] = 0

        return action_masks


    def reset(self, seed=None, options=None):
        super().reset(seed=42)

        self.service.reset()
        self.client.reset()
        self.current_step = 0
        # Initialize state with observations
        for i in range(self.history_size):
            self.state[i] = np.array([0, 0, 0, 0, 1/TOTAL_GPU,], dtype=np.float32)

        observation = self.state.flatten()
        return observation, {}

    def render(self, mode='human'):
        # print(f"Queue length: {self.queue_length}, Service instances: {self.service.num_inst}")
        return


