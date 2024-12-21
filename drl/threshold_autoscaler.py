def threshold_autoscaler_v2(sla_violation_rate: float, inst_util_usage: float, action_mask = None, sla_high = 0.001, sla_low = 0.0001, inst_util_high = 0.7, inst_util_low = 0.3, service=None):
    if service == 'ollama':
        sla_high, sla_low, inst_util_high, inst_util_low = 0.081, 0.0007, 0.90, 0.32

    elif service == 'pytorch':
        sla_high, sla_low, inst_util_high, inst_util_low = 0.095, 0.0008, 0.9, 0.38

    elif service == 'coqui':
        sla_high, sla_low, inst_util_high, inst_util_low = 0.091, 0.001, 0.89, 0.22

    action = 0
    if sla_violation_rate > sla_high or inst_util_usage > inst_util_high:
        action = 1
    elif sla_violation_rate < sla_low and inst_util_usage < inst_util_low:
        action = -1

    # print(f"sla_violation_rate {sla_violation_rate}, inst_util_usage {inst_util_usage}, action {action}")

    return action + 2

def threshold_autoscaler_v1(sla_violation_rate: float, inst_util_usage: float, action_mask = None, value_high_2 = 0.9, value_high_1 = 0.7, value_low_1 = 0.3, value_low_2 = 0.1, service=None):
    if service == 'ollama':
        value_high_2, value_high_1, value_low_1, value_low_2 = 0.82, 0.8, 0.44, 0.2
    elif service == 'pytorch':
        value_high_2, value_high_1, value_low_1, value_low_2 = 0.93, 0.8, 0.47, 0.17
    elif service == 'coqui':
        value_high_2, value_high_1, value_low_1, value_low_2 = 0.85, 0.8, 0.31, 0.14

    # action mask is handled by env itself
    action = 0
    if inst_util_usage > value_high_2:
        action = 2
    elif inst_util_usage > value_high_1:
        action = 1
    elif inst_util_usage < value_low_2:
        action = -2
    elif inst_util_usage < value_low_1:
        action = -1


    # map from -2->2 to 0->4 to match the env step interface
    action = action + 2

    return action

