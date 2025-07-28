#!/bin/bash

MAX_SAMPLES=15000
TOTAL_THREADS=10
NUM_THREADS=0

python3.11 ./main.py --max-samples $MAX_SAMPLES

while [ $NUM_THREADS -lt $TOTAL_THREADS ]
do
  python3.11 ./main.py --index-to-run $NUM_THREADS --total-executions $TOTAL_THREADS &

  NUM_THREADS=$((NUM_THREADS+1))
done

wait
