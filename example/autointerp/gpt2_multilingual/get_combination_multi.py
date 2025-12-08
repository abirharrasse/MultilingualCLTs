#!/usr/bin/env python3
import sys

# Configuration
MAX_LAYER = 12
TOTAL_JOBS = 1

# Generate all combinations
combinations = []
for layer in range(MAX_LAYER + 1):
    for job in range(TOTAL_JOBS):
        combinations.append((layer, job))

# Get the specific combination for this HTCondor job
job_index = int(sys.argv[1]) 

if job_index < len(combinations):
    layer, job = combinations[job_index]
    print(f"{layer} {job} {TOTAL_JOBS} {job_index}")
