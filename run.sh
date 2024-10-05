DATASET=bigcodebench
MODEL=meta-llama/Meta-Llama-3.1-8B-Instruct
BACKEND=vllm
NUM_GPU=2
SPLIT=complete
SUBSET=hard

bigcodebench.evaluate \
  --model $MODEL \
  --split $SPLIT \
  --subset $SUBSET \
  --backend $BACKEND \
  --tp $NUM_GPU