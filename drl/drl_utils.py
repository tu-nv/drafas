import json

SUCCESS = 0
ERROR = 1

# Constants
TOTAL_GPU = 8

USAGE_AVG_PERIOD_SEC = 30
INST_UTIL_LOG_UPDATE_PERIOD = 1000 # time resolution of update gpu usage log
USAGE_AVG_PERIOD_MS = USAGE_AVG_PERIOD_SEC * 1000  # Time units over which to average GPU usage
INST_UTIL_LOG_LEN = int(USAGE_AVG_PERIOD_MS/INST_UTIL_LOG_UPDATE_PERIOD)
SCALE_DOWN_COOLDOWN_STEP = 6

# service_configs = {
#         # (init_instances, gpu_per_inst, capacity_per_inst, proc_time_gpu in ms, req scaling, rate_limit_req_per_duration, rate_limit_duration),
#         # 'triton': (1, 1, 1, 124, 500, 36000, 0.2),
#         'pytorch': (1, 1, 1, 160, 500, 11000, 0.25, 6, 1000),
#         # 'ollama': (1, 1, 1, 440, 2500, 12000, 0.1),
#         'ollama': (1, 1, 1, 460, 2500, 11000, 0.1, 4, 2000),
#         'coqui': (1, 1, 1, 480, 2500, 21000, 0.1, 4, 2000),
#         # 'whisper': (1, 1, 1, 470, 1000, 14000, 0.1),
# }

service_configs = {
        # (init_instances, gpu_per_inst, capacity_per_inst, proc_time_gpu in ms, req scaling, rate_limit_req_per_duration, rate_limit_duration),
        'pytorch': (1, 1, 1, (60, 80), 500, 11000, 0.4, 14, 1000), # 70
        'ollama': (1, 1, 1, (140, 420), 2500, 13000, 0.15, 7, 2000), # 280
        'coqui': (1, 1, 1, (130, 390), 2500, 21000, 0.15, 7, 2000), #(260 + 400)//2
}

ENVOY_INBOUND = {
    'triton': "inbound_0.0.0.0_8000",
    'pytorch': "inbound_0.0.0.0_8005",
    'ollama': "inbound_0.0.0.0_11434",
    'coqui': "inbound_0.0.0.0_5002",
    'whisper': "inbound_0.0.0.0_9000",
}

CUSTOM_TEST_DIR = {
    'ollama': "shift_0h",
    'pytorch': "shift_8h",
    'coqui': "shift_16h",
}

class EvalStats():
    def __init__(self) -> None:
        self.sla_violation_rates = []
        self.sla_violation_rate_norms = []
        self.num_instances = []
        self.inst_utils = []
        self.avg_proc_times = []

    def add_observation(self, obs, info):
        sla_violation_rate = float(obs[-2])
        # sla_violation_rate_norm = float(obs[-2])
        inst_util = float(obs[-4])
        num_instance = float(obs[-1])
        avg_proc_time = float(info['avg_proc_time'])

        self.sla_violation_rates.append(sla_violation_rate)
        self.num_instances.append(int(num_instance * TOTAL_GPU))
        self.inst_utils.append(inst_util)
        self.avg_proc_times.append(avg_proc_time)
        # self.sla_violation_rate_norms.append(sla_violation_rate_norm)

    def write_stats(self, file_path):
        with open(file_path, 'w') as f:
            f.write(f"{self.summary_str()}\n"
                    f"sla_violation_rates: {self.sla_violation_rates}\n"
                    f"num_instances: {self.num_instances}\n"
                    f"avg_proc_times: {self.avg_proc_times}\n"
                    f"inst_utils: {self.inst_utils}\n")

    def summary(self):
        avg_sla_violation_rate = sum(self.sla_violation_rates) / len(self.sla_violation_rates)
        avg_num_instance = sum(self.num_instances) / len(self.num_instances)
        avg_proc_time = sum(self.avg_proc_times) / len(self.avg_proc_times)
        avg_inst_util = sum(self.inst_utils) / len(self.inst_utils)
        # print(f"Avg: slaviolation rate: {avg_sla_violation_rate:4f}, num instances: {avg_num_instance:4f}, inst_util: {avg_inst_util:4f}, proc time: {avg_proc_time:4f}")
        return avg_sla_violation_rate, avg_num_instance, avg_proc_time, avg_inst_util

    def summary_str(self):
        avg_sla_violation_rate, avg_num_instance, avg_proc_time, avg_inst_util = self.summary()
        # avg_sla_violation_rate_norm = sum(self.sla_violation_rate_norms)/len(self.sla_violation_rate_norms)
        num_inst_norm = avg_num_instance/TOTAL_GPU
        reward = -(0.9*avg_sla_violation_rate + 0.1*num_inst_norm)
        return f"Avg: sla_violation_rate: {avg_sla_violation_rate:.4f}, num_inst: {avg_num_instance:.4f}, inst_util: {avg_inst_util:.4f}, proc_time: {avg_proc_time:.4f}, reward: {reward:.5f}"

    def clear(self):
        self.sla_violation_rates = []
        self.num_instances = []
        self.inst_utils = []
        self.avg_proc_times = []
