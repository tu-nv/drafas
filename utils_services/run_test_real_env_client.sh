#!/bin/bash
set -x
set -e

PIDS=()
source .env/bin/activate

cleanup() {
    for pid in "${PIDS[@]}"; do
        kill "$pid" 2>/dev/null
    done
    wait # Wait for all processes to exit
}

trap cleanup SIGINT

# taskset -c 3-5 python drl/drl.py --mode real_env --service all --alg drl
# PIDS+=($!)
# sleep 0.5
taskset -c 0 python client/client.py --plan_dir trace/test/shift_0h --test_case ollama &
PIDS+=($!)
sleep 1
taskset -c 1 python client/client.py --plan_dir trace/test/shift_8h --test_case pytorch &
PIDS+=($!)
sleep 1
taskset -c 2 python client/client.py --plan_dir trace/test/shift_16h --test_case coqui &
PIDS+=($!)
sleep 1

wait
