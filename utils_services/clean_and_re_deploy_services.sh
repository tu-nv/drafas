#!/bin/bash
set -x

kubectl delete deployment ollama-deployment
sleep 1
kubectl delete deployment pytorch-deployment
sleep 1
kubectl delete deployment coqui-deployment
sleep 1

kubectl apply -f ai_services/ollama-deployment.yaml
sleep 1
kubectl apply -f ai_services/pytorch-deployment.yaml
sleep 1
kubectl apply -f ai_services/coqui-deployment.yaml
sleep 1

# kubectl scale deployment ollama-deployment --replicas=2
# sleep 1
# kubectl scale deployment pytorch-deployment --replicas=2
# sleep 1
# kubectl scale deployment whisper-deployment --replicas=2
