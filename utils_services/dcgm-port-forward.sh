#!/bin/bash
set -x
BASE_PORT=9401
PIDS=()

cleanup() {
    for pid in "${PIDS[@]}"; do
        kill "$pid" 2>/dev/null
    done
    wait # Wait for all processes to exit
}

trap cleanup SIGINT

for pod in $(kubectl -n gpu-operator get pod -o jsonpath='{.items[*].metadata.name}' | tr ' ' '\n' | grep nvidia-dcgm-exporter); do
    kubectl -n gpu-operator port-forward pod/$pod $BASE_PORT:9400 --address=0.0.0.0 &
    PIDS+=($!)
    BASE_PORT=$(( $BASE_PORT + 1 ))
done

wait
