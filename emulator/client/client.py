import asyncio
import aiohttp
import time
import os
import argparse
from utils import preprocess_img
from PIL import Image
import requests

BASE_DIR = os.path.dirname(os.path.realpath(__file__))

async def schedule_requests(request_data, num_requests, results, failed_requests, session: aiohttp.ClientSession):
    tasks = []
    for _ in range(num_requests):
        task = asyncio.create_task(post_request(request_data, session, results, failed_requests))
        tasks.append(task)
    await asyncio.gather(*tasks)

async def post_request(request_data, session: aiohttp.ClientSession, results, failed_requests, timeout=60):
    start = time.time()
    try:
        timeout_obj = aiohttp.ClientTimeout(total=timeout)
        if request_data['file_content']:
            # form_data can be used only once, so recreate it for every request
            form_data = aiohttp.FormData()
            form_data.add_field('audio_file', request_data['file_content'], filename="eng_male.wav")
            data = form_data
        else:
            data = None
        async with session.post(request_data['url'], json=request_data['json'], data=data, timeout=timeout_obj) as response:
            response_dat = await response.text()
            # print(response_dat)
    except asyncio.TimeoutError:
        print(f"Request timed out after {timeout} seconds")
        failed_requests.append('Timeout')
    except Exception as e:
        print(f"Request failed: {e}")
        failed_requests.append(e)
    else:
        end = time.time()
        elapsed = end - start
        results.append(elapsed)

async def load_plan(file_path):
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

async def cal_results(results, failed_requests):
    if results:
        average_time = sum(results) / len(results)
        min_time = min(results)
        max_time = max(results)
        print(f"Number of success requests: {len(results)}")
        print(f"Average time: {average_time:.4f} seconds")
        print(f"Min time: {min_time:.4f} seconds")
        print(f"Max time: {max_time:.4f} seconds")
    else:
        print("No successful requests to report.")

    if failed_requests:
        print(f"{len(failed_requests)} requests failed.")
    else:
        print("No requests failed.")

async def main():
    parser = argparse.ArgumentParser(description='Asyncio HTTP Requests Script')
    # parser.add_argument('--url', type=str, default="http://141.223.124.62:8080", help='The URL to send requests to')
    parser.add_argument('--plan_file', type=str, default="plan.txt", help='relative path to the plan file')
    parser.add_argument('--test_case', type=str, default="whisper", choices=['ollama', 'triton', 'whisper', 'simulator'],
                        help='test case, choose between ollama, triton, whisper, simulator')
    parser.add_argument('--speed_up', type=int, default=1, help='speed up factor')
    args = parser.parse_args()

    request_data = {
        "url": None,
        "file_content": None,
        "json": None,
        "headers": None
    }

    if args.test_case == 'triton':
        request_data['url'] = "http://141.223.124.62:30800/v2/models/resnet50/infer"
        img = Image.open(f"{BASE_DIR}/data/cup.jpg")
        # preprocessed img, FP32 formated numpy array
        data = preprocess_img(img, "float32", 224, 224)
        # data = data.tobytes()
        request_data["json"] = {
                "inputs": [
                    {
                    "name": "data",
                    "shape": [1, 3, 224, 224],
                    "datatype": "FP32",
                    "data": data.tolist()
                    }
                ],
                "outputs": [
                    {
                    "name": "resnetv24_dense0_fwd",
                    "parameters": { "classification" : 3 }
                    }
                ]
            }
        # test inference
        # response = requests.post(request_data['url'], json=request_data["json"])
        # print(response.text)
        # exit(1)
    elif args.test_case == 'ollama':
        request_data['url'] = "http://141.223.124.62:30434/api/generate"
        request_data['json'] = {
                    "model": "llama3.2:1b-instruct-q4_K_M",
                    "prompt": "How many r in strawberry?",
                    "options": {
                        "num_gpu": 99
                    },
                        "stream": False
                }
    elif args.test_case == 'whisper':
        request_data['url'] = "http://141.223.124.62:30900/asr?output=json"
        request_data['headers'] = {'content-type': 'multipart/form-data'}
        with open(f"{BASE_DIR}/data/eng_male.wav", "rb") as audio_file:
            request_data['file_content'] = audio_file.read()
        # exit(0)
    elif args.test_case == 'simulator':
        request_data['url'] = "http://141.223.124.61:8101/process_request"
        request_data['json'] = {
                    "message": "hello simulator",
                }
    else:
        print(f"Wrong test case: {args.test_case}")
        exit(1)

    results = []
    failed_requests = []

    plan = await load_plan(f'{BASE_DIR}/{args.plan_file}')

    # Determine the maximum number of concurrent requests
    max_concurrency = max(num_requests for _, num_requests in plan)

    connector = aiohttp.TCPConnector(limit=max_concurrency)

    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = []

        for delay, num_requests in plan:
            task = asyncio.create_task(
                schedule_requests(request_data, num_requests, results, failed_requests, session)
            )
            tasks.append(task)
            await asyncio.sleep(delay)

        # Wait for all tasks to complete
        await asyncio.gather(*tasks)

    # Now calculate average, min, max times
    await cal_results(results, failed_requests)

if __name__ == '__main__':
    asyncio.run(main())
