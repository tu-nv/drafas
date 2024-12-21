#!/bin/bash
set -x

SEVICE="$1"
NAMESPACE="default"
DEPLOYMENT_NAME="$SEVICE-deployment"

kubectl delete deployment $DEPLOYMENT_NAME
sleep 8

start_time=$(date +%s.%N)
kubectl apply -f ai_services/$SEVICE-deployment.yaml
sleep 1

if [ "$SEVICE" = "ollama" ]; then
    curl --retry 60 --retry-connrefused --retry-delay 1 --fail http://141.223.124.62:30434/
elif [ "$SEVICE" = "triton" ]; then
    curl --retry 60 --retry-connrefused --retry-delay 1 --fail http://141.223.124.62:30800/v2/health/ready
elif [ "$SEVICE" = "pytorch" ]; then
    curl --retry 60 --retry-connrefused --retry-delay 1 --fail http://141.223.124.62:30805/ready
elif [ "$SEVICE" = "whisper" ]; then
    curl --retry 60 --retry-connrefused --retry-delay 1 --fail 141.223.124.62:30900/
elif [ "$SEVICE" = "coqui" ]; then
    curl --retry 60 --retry-connrefused --retry-delay 1 --fail http://141.223.124.62:30502/static/coqui-log-green-TTS.png
else
    echo "Unsupport service $SERVICE"
    exit 1
fi


end_time=$(date +%s.%N)

elapsed=$(echo "$end_time - $start_time" | bc)
echo "Startup time $elapsed seconds"
