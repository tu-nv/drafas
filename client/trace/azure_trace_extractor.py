import csv
import os
from datetime import datetime, timedelta
import pickle

BASE_DIR = os.path.dirname(os.path.realpath(__file__))

req_over_time = {}
with open(f'{BASE_DIR}/AzureLLMInferenceTrace_code_1week.csv', mode ='r') as file:
    csvFile = csv.reader(file)
    begin_ts = None
    firstline_discarded = False
    for line in csvFile:
        if not firstline_discarded:
            firstline_discarded = True
            continue
        timestamp = line[0]
        timestamp = datetime.fromisoformat(timestamp)

        current_date = str(timestamp.date())
        if current_date not in req_over_time:
            req_over_time[current_date] = [0 for _ in range(24*3600)]
        idx = timestamp.hour * 3600 + timestamp.minute * 60 + timestamp.second
        req_over_time[current_date][idx] += 1

avg_all = []
for date in req_over_time.keys():
    avg_req = sum(req_over_time[date])/len(req_over_time[date])
    print(f"date: {date}, avg: {avg_req}, min: {min(req_over_time[date])}, max: {max(req_over_time[date])}")
    avg_all.append(avg_req)
    with open(f'{BASE_DIR}/req_over_time_{date}.txt', 'w') as f:
        for x in req_over_time[date]:
            f.write(f"1 {x}\n")

print(f"overall avg: {sum(avg_all)/len(avg_all)}")
