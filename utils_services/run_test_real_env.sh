#!/bin/bash
set -x
set -e

ALG="$1"

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
taskset -c 3 python drl/drl.py --mode real_env --service ollama --alg $ALG > logs/$ALG-ollama.txt &
PIDS+=($!)
sleep 1
taskset -c 4 python drl/drl.py --mode real_env --service pytorch --alg $ALG > logs/$ALG-pytorch.txt &
PIDS+=($!)
sleep 1
taskset -c 5 python drl/drl.py --mode real_env --service coqui --alg $ALG > logs/$ALG-coqui.txt &
PIDS+=($!)
sleep 1

wait
