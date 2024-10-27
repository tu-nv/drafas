import asyncio
from fastapi import FastAPI, Body
import uvicorn
import time
from contextlib import asynccontextmanager


TYPE_GPU = 0
TYPE_CPU = 1

TOTAL_CPUS = 0  # cores
TOTAL_GPUS = 16  # MPS partitions
TOTAL_MEMORY = 102  # GBs

SPEED_UP = 1 # speedup simulation

class SharedResources:
    def __init__(self, total_cpus, total_gpus, total_memory):
        self.cpus_used = 0.0
        self.gpus_used = 0
        self.memory_used = 0.0
        self.total_cpus = total_cpus
        self.total_gpus = total_gpus
        self.total_memory = total_memory
        self.lock = asyncio.Lock()

shared_resources = SharedResources(TOTAL_CPUS, TOTAL_GPUS, TOTAL_MEMORY)

def create_app(cpu_per_inst, memory_per_inst, gpu_per_inst, capacity, proc_time_cpu, proc_time_gpu):
    app = FastAPI()

    class Instance:
        def __init__(self, id, capacity, type):
            self.id = id
            self.capacity = capacity
            self.type = type
            self.semaphore = asyncio.Semaphore(capacity)
            self.current_requests = 0  # For tracking purposes
            self.proc_time = proc_time_cpu if type == TYPE_CPU else proc_time_gpu

    instances = []
    instances_lock = asyncio.Lock()


    # Variables to track usage over time
    usage_history = []  # List of tuples (timestamp, cpu_usage_percent, gpu_usage_percent)
    last_usage_read_time = time.time()

    # Variables to track processing times
    processing_times = []  # List of processing times since last /usage call
    processing_times_lock = asyncio.Lock()  # Lock for processing_times list

    next_instance_id = 0  # To assign unique IDs to instances

    async def update_usage_history():
        total_cpu_usage = 0.0
        total_gpu_usage = 0.0
        async with instances_lock:
            for instance in instances:
                if instance.type == TYPE_CPU:
                    cpu_usage_per_instance = (instance.current_requests / instance.capacity)
                    total_cpu_usage += cpu_usage_per_instance
                elif instance.type == TYPE_GPU:
                    # Each GPU instance uses negligible CPU and memory that is reserved in the node already, so not count here
                    # Each GPU instance uses 1 mps partition
                    gpu_usage_per_instance = (instance.current_requests / instance.capacity)
                    total_gpu_usage += gpu_usage_per_instance

        # Calculate usage percentages
        cpu_usage_percent = total_cpu_usage / len(instances)
        gpu_usage_percent = total_gpu_usage / len(instances)

        usage_history.append((time.time(), cpu_usage_percent, gpu_usage_percent))

    # Automatically scale one instance at startup using lifespan
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        await scale(1)  # Scale up 1 instance at startup
        yield
        # Add any shutdown logic here if needed

    app.router.lifespan_context = lifespan

    @app.post("/scale")
    async def scale(change: int = Body(..., embed=True)):
        nonlocal instances, next_instance_id
        if change == 1:
            # Scale up by 1 instance
            async with instances_lock:
                # Try to add GPU instance if resources allow
                async with shared_resources.lock:
                    if (shared_resources.gpus_used + gpu_per_inst <= shared_resources.total_gpus):
                        # Add GPU instance
                        shared_resources.gpus_used += gpu_per_inst
                        type = TYPE_GPU
                    elif (shared_resources.cpus_used + cpu_per_inst <= shared_resources.total_cpus and
                          shared_resources.memory_used + memory_per_inst <= shared_resources.total_memory):
                        # Add CPU instance
                        shared_resources.cpus_used += cpu_per_inst
                        shared_resources.memory_used += memory_per_inst
                        type = TYPE_CPU
                    else:
                        message = {"status": "failure", "message": "Not enough resources to scale up"}
                        print(message)
                        return message
                # Create and add the instance
                new_instance = Instance(next_instance_id, capacity, type)
                next_instance_id += 1
                instances.append(new_instance)
            await update_usage_history()
            return {
                "status": "success",
                "action": "scaled up",
                "type": "GPU" if type == TYPE_GPU else "CPU",
                "instance_id": new_instance.id
            }
        elif change == -1:
            # Scale down by 1 instance
            async with instances_lock:
                if len(instances)==0:
                    message = {"status": "failure", "message": "No instances to scale down"}
                    print(message)
                    return message
                # Remove an instance
                instance_to_remove = instances.pop()
                # Release resources
                async with shared_resources.lock:
                    if instance_to_remove.type == TYPE_GPU:
                        shared_resources.gpus_used -= gpu_per_inst
                    elif instance_to_remove.type == TYPE_CPU:
                        shared_resources.cpus_used -= cpu_per_inst
                        shared_resources.memory_used -= memory_per_inst
                    else:
                        message = {"status": "failure", "message": "Wrong instance type"}
                        print(message)
                        return message
            await update_usage_history()
            return {
                "status": "success",
                "action": "scaled down",
                "type": "GPU" if instance_to_remove.type == TYPE_GPU else "CPU",
                "instance_id": instance_to_remove.id
            }
        elif change == 0:
            # Maintain current number of instances
            return {"status": "success", "action": "no change"}
        else:
            message = {"status": "failure", "message": "Invalid scale parameter"}
            print(message)
            return message

    async def acquire_slot():
        nonlocal instances
        while True:
            async with instances_lock:
                for instance in instances:
                    if instance.current_requests < instance.capacity:
                        # Acquire a slot
                        await instance.semaphore.acquire()
                        instance.current_requests += 1
                        return instance
            # If no instances have capacity, wait a bit and retry
            await asyncio.sleep(0.001)
        #

    @app.post("/process_request")
    async def process_request():
        nonlocal processing_times, processing_times_lock
        instance = await acquire_slot()
        await update_usage_history()
        start_time = time.time()

        try:
            # Simulate processing time
            await asyncio.sleep(instance.proc_time / (1000.0 * SPEED_UP))
            end_time = time.time()
            actual_processing_time = (end_time - start_time) * 1000 * SPEED_UP # Convert to milliseconds
            return {
                "status": "processed",
                "instance_id": instance.id,
                "type": "GPU" if instance.type == TYPE_GPU else "CPU",
                "processing_time_ms": actual_processing_time
            }
        finally:
            async with processing_times_lock:
                processing_times.append(actual_processing_time)
            # Release the slot and decrement current_requests
            async with instances_lock:
                instance.current_requests -= 1
                instance.semaphore.release()
            await update_usage_history()

    @app.get("/stats")
    async def get_stats():
        nonlocal last_usage_read_time, usage_history, processing_times
        now = time.time()

        # Initialize variables
        total_cpu_usage_time = 0.0
        total_gpu_usage_time = 0.0
        total_time = now - last_usage_read_time

        # Make a copy of usage_history to prevent modification during iteration
        usage_history_copy = usage_history.copy()

        # Initialize prev_time and prev_cpu_usage, prev_gpu_usage
        prev_time = last_usage_read_time
        prev_cpu_usage = None
        prev_gpu_usage = None

        # Find the last usage level before last_usage_read_time
        for timestamp, cpu_usage_percent, gpu_usage_percent in reversed(usage_history_copy):
            if timestamp <= last_usage_read_time:
                prev_cpu_usage = cpu_usage_percent
                prev_gpu_usage = gpu_usage_percent
                break

        # If we didn't find any usage level before last_usage_read_time, use the first recorded usage
        if prev_cpu_usage is None or prev_gpu_usage is None:
            if usage_history_copy:
                prev_cpu_usage = usage_history_copy[0][1]
                prev_gpu_usage = usage_history_copy[0][2]
            else:
                prev_cpu_usage = 0.0
                prev_gpu_usage = 0.0

        # Now process the usage_history entries
        for timestamp, cpu_usage_percent, gpu_usage_percent in usage_history_copy:
            if timestamp < last_usage_read_time:
                continue
            duration = timestamp - prev_time
            total_cpu_usage_time += prev_cpu_usage * duration
            total_gpu_usage_time += prev_gpu_usage * duration
            prev_time = timestamp
            prev_cpu_usage = cpu_usage_percent
            prev_gpu_usage = gpu_usage_percent

        # Account for time from the last recorded change to now
        duration = now - prev_time
        total_cpu_usage_time += prev_cpu_usage * duration
        total_gpu_usage_time += prev_gpu_usage * duration

        avg_cpu_usage = (total_cpu_usage_time / total_time) if total_time > 0 else 0.0
        avg_gpu_usage = (total_gpu_usage_time / total_time) if total_time > 0 else 0.0

        # Update last_usage_read_time
        last_usage_read_time = now

        # Clean up usage_history to remove entries before last_usage_read_time
        usage_history[:] = [(t, cpu, gpu) for t, cpu, gpu in usage_history if t >= last_usage_read_time]

        # Calculate average processing time
        async with processing_times_lock:
            if processing_times:
                avg_processing_time = sum(processing_times) / len(processing_times)
            else:
                avg_processing_time = 0.0
            # Clear the processing_times list
            processing_times.clear()

        # Get remaining resources
        async with shared_resources.lock:
            remaining_cpus = shared_resources.total_cpus - shared_resources.cpus_used
            remaining_gpus = shared_resources.total_gpus - shared_resources.gpus_used
            remaining_memory = shared_resources.total_memory - shared_resources.memory_used

        # Count instances by type
        async with instances_lock:
            num_gpu_instances = sum(1 for instance in instances if instance.type == TYPE_GPU)
            num_cpu_instances = sum(1 for instance in instances if instance.type == TYPE_CPU)

        return {
            "cpu_usage": avg_cpu_usage,
            "gpu_usage": avg_gpu_usage,
            "average_processing_time_ms": avg_processing_time,
            "remaining_cpus": remaining_cpus,
            "remaining_gpus": remaining_gpus,
            "remaining_memory": remaining_memory,
            "num_instances": len(instances),
            "num_gpu_instances": num_gpu_instances,
            "num_cpu_instances": num_cpu_instances
        }

    return app

async def run_multiple_servers(configs):
    servers = []
    for config in configs:
        cpu_per_inst, memory_per_inst, gpu_per_inst, capacity, proc_time_cpu, proc_time_gpu, port = config
        app = create_app(cpu_per_inst, memory_per_inst, gpu_per_inst, capacity, proc_time_cpu, proc_time_gpu)
        config = uvicorn.Config(app, host="0.0.0.0", port=port, log_level="info")
        server = uvicorn.Server(config)
        servers.append(server.serve())
    await asyncio.gather(*servers)

if __name__ == "__main__":
    configs = [
        # (cpu_per_inst, memory_per_inst, gpu_per_inst, capacity, proc_time_cpu, proc_time_gpu, port),
        # ollama
        (2, 4, 1, 2, 1000, 560, 8101),
        # Add more configurations as needed
    ]
    asyncio.run(run_multiple_servers(configs))
