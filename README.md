# DRAFAS
Dynamic Resource Allocation For AI-native Services (DRAFAS).

## Cluster installation
We need a cluster with at least one server and one Nvidia GPU.
- Install K3s following [instruction](https://docs.k3s.io/installation).
- Install [NVIDIA gpu operator](https://github.com/NVIDIA/gpu-operator) for k3s
   ```bash
   helm install --wait nvidiagpu -n gpu-operator --create-namespace \
      --values utils_services/gpu-operator-values.yaml \
      nvidia/gpu-operator

   ```
- GPU slicing with MPS or time-slicing. By default, we use MPS. If you want to use time-slicing, you need to modify `utils_services/mps-slicing-config-all.yaml`
   ```bash
   kubectl create -n gpu-operator -f utils_services/mps-slicing-config-all.yaml
   kubectl patch clusterpolicies.nvidia.com/cluster-policy \
      -n gpu-operator --type merge \
      -p '{"spec": {"devicePlugin": {"config": {"name": "mps-slicing-config-all", "default": "any"}}}}'
   # check if patch working correctly
   kubectl get events -n gpu-operator --sort-by='.lastTimestamp'
   ```
- Install Prometheus
   ```bash
   kubectl apply -f utils_services/prometheus-deployment.yaml
   ```

- Install Istio following [instruction](https://istio.io/latest/docs/setup/install/). Then config istio to only expose neccessary metrics.
   ```bash
   istioctl install -f utils_services/istio-operator.yaml
   ```

## Service deployment
We provide three prototype AI services for evaluation: a chatbot service (ollama), an image classification service (pytorch), and a text to speech service (coqui). To deploy them, run bellow commands from K3s manager node:
   ```bash
   kubectl apply -f ai_services/ollama-deployment.yaml
   kubectl apply -f ai_services/pytorch-deployment.yaml
   kubectl apply -f ai_services/coqui-deployment.yaml
   ```

## Auto scaling agent
The auto scaling agent (DRL-based or rule-based) should be deployed in a separated machine (no GPU is required).

- A virtual environment should be used.
   ```bash
   cd drafas
   python3.10 -m venv .env
   source .env/bin/activate
   pip install -r requirement.txt
   ```
- Running the agent
   ```bash
   # this command start all three agents. To manually start each agent, check the script for details.
   ./utils_services/run_test_real_env.sh drl
   # if use rule-based agent, use bellow command
   # ./utils_services/run_test_real_env.sh th

   # for custom command, check help message
   python drl/drl.py --help
   ```

## Client emulator
The client emulator send inference requests to the AI services. If you run the client in the different python env than the one for auto scaling agent, you need to first prepair the python virtual env similar to the auto scaling agent.
- Generate request content dataset.
   ```bash
   python client/generate_data.py
   ```
- Start sending request to AI services from client emulator.
   ```bash
   ./utils_services/run_test_real_env_client.sh drl
   # for custom client command, see help
   python client/client.py --help
   ```
## Network slicing
In DRAFAS, network slicing is optional and can be created using 5G/B5G network slicing. For fast testing, two simple network slices can be created using helm chart from [gradiant repo](https://github.com/Gradiant/5g-charts), following the [tutorial](https://gradiant.github.io/5g-charts/open5gs-ueransim-gnb.html). The slice's parameters can be setup via open5GS web interface.

## Simulation
For fast training and evaluation, we implemented a simulator in `simpy_env.py`. For training in simulator:
```bash
taskset -c 2-3 python drl/drl.py --mode train --service ollama
# for custom train command, see help
python drl/drl.py --help
```

## Publication
To understand how DRAFAS works, please check our publication at `to be updated`
