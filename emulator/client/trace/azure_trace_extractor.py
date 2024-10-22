import csv
import os
from datetime import datetime, timedelta

BASE_DIR = os.path.dirname(os.path.realpath(__file__))

with open(f'{BASE_DIR}/AzureLLMInferenceTrace_code_1week.csv', mode ='r') as file:
    csvFile = csv.reader(file)
    first_ts = None
    req_cnt = 1
    firstline_discarded = False
    for line in csvFile:
        if not firstline_discarded:
            firstline_discarded = True
            continue
        timestamp = line[0]
        timestamp = datetime.fromisoformat(timestamp)

        if not first_ts:
            first_ts = timestamp
            continue
        if timestamp - first_ts < timedelta(seconds=60):
            req_cnt += 1
            continue
        print(req_cnt//60)
        first_ts = timestamp
        req_cnt = 1
