DATASET=bigcodebench
MODEL=meta-llama/Llama-3.2-1B-Instruct
BACKEND=vllm
NUM_GPU=1
SPLIT=complete
SUBSET=hard

bigcodebench.evaluate \
  --model $MODEL \
  --samples meta-llama--Llama-3.2-1B-Instruct--bigcodebench-hard-complete--vllm-0-1-sanitized_calibrated.jsonl \
  --split $SPLIT \
  --subset $SUBSET \
  --backend $BACKEND \
  --greedy