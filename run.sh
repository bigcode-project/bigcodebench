DATASET=bigcodebench
MODEL=meta-llama/Meta-Llama-3.1-8B-Instruct
BACKEND=vllm
NUM_GPU=2
SPLIT=complete
SUBSET=full
export E2B_API_KEY="e2b_0a231fa3b0a2b01690ab6c66a23b55c0979ce4ee"

bigcodebench.evaluate \
  --samples bcb_results/gemini-2.0-flash-exp--main--bigcodebench-instruct--google-0-1-sanitized_calibrated.jsonl \
  --model $MODEL \
  --split $SPLIT \
  --subset $SUBSET \
  --backend $BACKEND \
  --check_gt_only