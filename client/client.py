import asyncio
import aiohttp
import time
import os, io
import argparse
from client_utils import preprocess_img, create_random_audio
import uvicorn
from fastapi import FastAPI
from urllib.parse import quote
import random, pickle
from generate_data import sentences
import sys
from datetime import datetime
import pytz

# Define the Seoul timezone
seoul_timezone = pytz.timezone('Asia/Seoul')

BASE_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.append(f'{BASE_DIR}/../drl')
from drl_utils import service_configs

random.seed(42)


# service_configs = {
#         # for triton, somehow delay increase 80ms compared to report in istio, so plus 80 here 500+80
#         # (init_instances, gpu_per_inst, capacity_per_inst, proc_time_gpu in ms, req scaling),
#         'triton': (1, 1, 1, 124, 500, 36000, 0.3),
#         'ollama': (1, 1, 1, 440, 2500, 11000, 0.1),
#         # 'whisper': (1, 1, 1, 280, 1000, 14000, 0.15),
#         'coqui': (1, 1, 1, 490, 2500, 21000, 0.1),
#         'pytorch': (1, 1, 1, 150, 500, 5000, 0.2),
# }

# data for testing

# audio_samples = load_google_speech_command_dataset()
audio_samples = []

app = FastAPI()

results = []
results_lock = asyncio.Lock()

# for calculate stats
sum_proc_time = 0.0
num_success_req = 0
num_sla_violated_req = 0
num_failed_req = 0

avg_sla_violation_rate_arr = []
avg_proc_time_arr = []

start_time = time.monotonic()


parser = argparse.ArgumentParser(description='Asyncio HTTP Requests Script')
# parser.add_argument('--url', type=str, default="http://141.223.124.62:8080", help='The URL to send requests to')
parser.add_argument('--plan_dir', type=str, default="trace/test", help='relative path to the trace dir')
parser.add_argument('--test_case', type=str, default="ollama",
                    help='test case, choose between ollama, coqui, pytorch')
parser.add_argument('--speedup', type=int, default=1, help='speed up factor')
parser.add_argument('--start_point', type=int, default=0, help='start from this timestamp point')
parser.add_argument('--print_response', default=False, action="store_true", help='print request response or not')
parser.add_argument('--constant_rate', type=int, default=None, help='constant request rates, disregard plan_dir')
parser.add_argument('--continuous_request', default=False, action="store_true", help='constant sending succeeded request one imediately after one')
parser.add_argument('--instant_request', default=False, action="store_true", help='instant request when new second start (i.e., do not distribute request evenly over period)')
# parser.add_argument('--amplification', type=int, default=1, help='amplification factor for increasing or decreasing number of req')
args = parser.parse_args()


request_data = {
        "url": None,
        "json": None,
        "headers": None
    }


delay_sla = service_configs[args.test_case][4] / 1000
req_multiply = service_configs[args.test_case][6]


# req_multiply set to 2*(num parallel req per instance) / (avg req per second of the trace)
if args.test_case == 'triton':
    with open(f"{BASE_DIR}/data/image_dataset.pkl", "rb") as f:
        image_dataset = pickle.load(f)
    stats_port = 8201
    request_data['url'] = "http://141.223.124.62:30800/v2/models/resnet152/infer"
    # img = Image.open(f"{BASE_DIR}/data/cup.jpg")
    # # preprocessed img, FP32 formated numpy array
    # data = preprocess_img(img, "float32", 224, 224)
    # data = data.tobytes()
    request_data["json"] = {
            "inputs": [
                {
                "name": "data",
                "shape": [1, 3, 224, 224],
                "datatype": "FP32",
                # "data": data.tolist()
                }
            ],
            "outputs": [
                {
                "name": "resnetv27_dense0_fwd",
                "parameters": { "classification" : 1 }
                }
            ]
        }
    # test inference
    # response = requests.post(request_data['url'], json=request_data["json"])
    # print(response.text)
    # exit(1)

elif args.test_case == 'ollama':
    with open(f"{BASE_DIR}/data/llm_questions_data.pkl", "rb") as f:
        llm_questions = pickle.load(f)
    stats_port = 8202
    request_data['url'] = "http://141.223.124.63:30434/api/generate"
    request_data['json'] = {
                "model": "llama3.2:1b-instruct-q4_K_M",
                "options": {
                    "num_gpu": 99
                },
                    "stream": False
            }

elif args.test_case == 'whisper':
    with open(f"{BASE_DIR}/data/audio_data.pkl", "rb") as f:
        audio_data = pickle.load(f)
    stats_port = 8203
    request_data['url'] = "http://141.223.124.62:30900/asr?output=json"

    # for file in os.listdir(f"{BASE_DIR}/data/sound/"):
    #     with open(f"{BASE_DIR}/data/sound/{file}", "rb") as audio_file:
    #         audio_samples.append(audio_file.read())

    with open(f"{BASE_DIR}/data/eng_male.wav", "rb") as audio_file:
        audio_sample = audio_file.read()

elif args.test_case == 'coqui':
    stats_port = 8204
    request_data['url'] = "http://141.223.124.62:30502/api/tts"

elif args.test_case == 'pytorch':
    with open(f"{BASE_DIR}/data/image_dataset_pytorch.pkl", "rb") as f:
        image_dataset_pytorch = pickle.load(f)
    stats_port = 8205
    # request_data['url'] = "http://141.223.124.61:8005/classify_batch"
    request_data['url'] = "http://141.223.124.62:30805/classify_batch"
else:
    print(f"Wrong test case: {args.test_case}")
    exit(1)


async def schedule_requests(request_data, num_requests, delay_bw_request, session: aiohttp.ClientSession):
    tasks = []
    for i in range(num_requests):
        delay_before_request = i * delay_bw_request
        task = asyncio.create_task(post_request(delay_before_request, request_data, session))
        tasks.append(task)
    await asyncio.gather(*tasks)

async def post_request(delay_before_request, request_data, session: aiohttp.ClientSession, timeout=4):
    global sum_proc_time, num_success_req, num_failed_req, num_sla_violated_req
    if not args.instant_request:
        await asyncio.sleep(delay_before_request)
        # print(f"delayed {delay_before_request}")
    start = time.monotonic()
    try:
        timeout_obj = aiohttp.ClientTimeout(total=timeout)
        query_params = None
        data = None

        if args.test_case == "whisper":
            audio = random.choice(audio_data)
            # audio = audio_data[100]
            # audio = create_random_audio()
            # form_data can be used only once, so recreate it for every request
            # audio = audio_sample
            form_data = aiohttp.FormData()
            form_data.add_field('audio_file', audio, filename="audio.wav")
            query_params = {"language": "en", "encode": 'False'}
            data = form_data

        elif args.test_case == "coqui":
            sentence = random.choice(sentences)
            query_params = {
                                "text": sentence,
                                "speaker_id": "p376"
                            }
        elif args.test_case == "ollama":
            request_data['json']['prompt'] = f"{random.choice(llm_questions)}. Answer less than 10 words."
            # request_data['json']['cache_prompt'] = False
            # request_data['url'] = random.choice(["http://141.223.124.62:30434/api/generate", "http://141.223.124.63:30434/api/generate"])

        elif args.test_case == "triton":
            image_data = random.choice(image_dataset)
            request_data['json']['inputs'][0]['data'] = image_data

        elif args.test_case == "pytorch":
            # image_data = random.choice(image_dataset_pytorch)
            # buffer = io.BytesIO()
            # image_data.save(buffer, format="JPEG")  # Save directly since it's a PIL image
            # buffer.seek(0)

            batch_size = 4
            form_data = aiohttp.FormData()
            for idx in range(0, batch_size):
                image_data = random.choice(image_dataset_pytorch)
                buffer = io.BytesIO()
                image_data.save(buffer, format="JPEG")  # Save directly since it's a PIL image
                buffer.seek(0)
                form_data.add_field(
                    "files",
                    buffer,  # The binary data of the image
                    filename=f"cifar10_{idx}.jpg",  # File name
                    content_type="image/jpeg",  # Content type of the image
                )
            data = form_data

        headers = {'Connection': 'close'}  # Close the connection after each request
        start = time.monotonic()
        # sending request
        if args.test_case == "coqui":
            async with session.get(request_data['url'], params=query_params, headers=headers, timeout=timeout_obj) as response:
                status_code = response.status
                response_dat = await response.read()
                if args.print_response:
                    print(f"audio size: {len(response_dat)}")
                    if (len(response_dat) < 100):
                        print(status_code, response_dat)
        else:
            async with session.post(request_data['url'], json=request_data['json'], data=data, params=query_params, headers=headers, timeout=timeout_obj) as response:
            # async with session.get(request_data['url']) as response:
                status_code = response.status
                response_dat = await response.text()
                if args.print_response:
                    print(response_dat)

    except asyncio.TimeoutError:
        # print(f"Request timed out after {timeout} seconds")
        num_failed_req += 1
    except Exception as e:
        print(f"Request failed: {e}")
        num_failed_req += 1
    else:
        if status_code != 200:
            # print(f"Request failed with status code {status_code}")
            num_failed_req += 1
        else:
            processing_time = (time.monotonic() - start) * args.speedup
            sum_proc_time += processing_time
            num_success_req += 1
            if processing_time > delay_sla:
                num_sla_violated_req += 1
            async with results_lock:
                results.append([time.monotonic(), processing_time])
        # else:
        #     num_failed_req += 1

async def load_plan_from_file(file_path):
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

async def send_reqs_with_plan(connector, plan):
    sample_cnt = 0
    async with aiohttp.ClientSession(connector=connector) as session:
        # tasks = []

        for delay, num_requests in plan:
            sample_cnt += 1
            if args.continuous_request:
                await post_request(0, request_data, session)
            else:
                if not args.constant_rate:
                    num_requests = int(num_requests*req_multiply)
                    # print(f'num_requests: {num_requests}')
                delay = delay/args.speedup
                if num_requests > 0:
                    task = asyncio.create_task(
                        schedule_requests(request_data, int(num_requests), delay/num_requests, session)
                    )
                    # tasks.append(task)
                await asyncio.sleep(delay)

            if(sample_cnt % 30 == 0):
                print(f'---------------{args.test_case}----------------')
                await cal_and_print_results()

        # Wait for all tasks to complete
        # await asyncio.gather(*tasks)

async def exec_plan():
    # Determine the maximum number of concurrent requests
    max_concurrency = 1000

    connector = aiohttp.TCPConnector(limit=max_concurrency)

    if not args.constant_rate and not args.continuous_request:
        for f in os.listdir(f'{BASE_DIR}/{args.plan_dir}'):
            plan_file = os.path.join(f'{BASE_DIR}/{args.plan_dir}', f)
            plan = await load_plan_from_file(plan_file)
            print(f"Start with plan file {plan_file}")
            await send_reqs_with_plan(connector, plan)
    else:
        plan = [(1, args.constant_rate) for _ in range (100)]
        await send_reqs_with_plan(connector, plan)


async def cal_and_print_results():
    global sum_proc_time, num_sla_violated_req, num_failed_req, num_success_req

    avg_sla_violation_rate = (num_sla_violated_req + num_failed_req)/(num_success_req + num_failed_req + 0.001)
    avg_sla_violation_rate_wo_failed_req = num_sla_violated_req/(num_success_req + 0.001)
    avg_proc_time = sum_proc_time/(num_success_req + 0.001)

    avg_sla_violation_rate_arr.append(avg_sla_violation_rate)
    avg_proc_time_arr.append(avg_proc_time)

    print(
        f"Service {args.test_case}: Time: {int(time.monotonic() - start_time)} "
        f"Number of success requests: {num_success_req}\n"
        f"Average time: {avg_proc_time:4f} seconds\n"
        f"SLA violation rate: {avg_sla_violation_rate:4f}\n"
        f"Number of failed reqests: {num_failed_req}\n"
        f"SLA violation rate w/o failed req: {avg_sla_violation_rate_wo_failed_req:4f} seconds\n"
        f"sla violation all time: {sum(avg_sla_violation_rate_arr)/(len(avg_sla_violation_rate_arr) + 0.0001)}, "
        f"avg proc time all time: {sum(avg_proc_time_arr)/(len(avg_proc_time_arr) + 0.0001)}, \n"
        # f"avg_sla_violation_rate_arr: {avg_sla_violation_rate_arr}\n"
        # f"avg_proc_time_arr: {avg_proc_time_arr}\n"
    )

    num_sla_violated_req = 0
    num_failed_req = 0
    num_success_req = 0
    sum_proc_time = 0.0





@app.get("/stats")
async def get_stats():
    global results
    now = time.monotonic()
    sla_violation_rate = 0.0
    avg_processing_time_s = 0.0
    async with results_lock:
        if len(results) > 0:
            # only take results of last 15 secs, otherwise the results will grow too large overtime
            recent_results = [x for x in results if (now - x[0]) * args.speedup <= 15]
            results = recent_results
            recent_proc_times = [x[1] for x in results]
            if len(recent_proc_times) > 0:
                sla_violations = [x for x in recent_proc_times if x > delay_sla]
                sla_violation_rate = len(sla_violations) / len(recent_proc_times)
                avg_processing_time_s = sum(recent_proc_times) / len(recent_proc_times)

    return {
        "sla_violation_rate": sla_violation_rate,
        "avg_processing_time_s": avg_processing_time_s
    }

async def write_stats(file_path):
    with open(file_path, 'w') as f:
        f.write(f"{args.test_case}\n"
                f"sla_violation_rates: {avg_sla_violation_rate_arr}\n"
                f"avg_proc_times: {avg_proc_time_arr}\n"
        )


async def run_app():
    config = uvicorn.Config(app=app, host="0.0.0.0", port=stats_port, loop="asyncio", lifespan="off", log_level="warning")
    server = uvicorn.Server(config)
    # Run the server
    await server.serve()

async def main():
    server_task = asyncio.create_task(run_app())
    await asyncio.sleep(1)
    try:
        day_cnt = 0
        while True:
            await asyncio.sleep(1)
            await exec_plan()
            day_cnt += 1
            print(f"--------Finish day {day_cnt}--------")

    finally:
        await cal_and_print_results()
        # Define the Seoul timezone

        # Get the current datetime in the Seoul timezone
        now = datetime.now(seoul_timezone)
        await write_stats(f'{BASE_DIR}/../stats/real-env-client_{now}.txt')
        server_task.cancel()

if __name__ == '__main__':
    asyncio.run(main())
