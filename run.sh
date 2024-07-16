BS=5
DATASET=bigcodebench
MODEL=gpt-3.5-turbo-0125
BACKEND=openai
TEMP=0
N_SAMPLES=1
NUM_GPU=1
SPLIT=complete
SUBSET=hard
if [[ $MODEL == *"/"* ]]; then
  ORG=$(echo $MODEL | cut -d'/' -f1)--
  BASE_MODEL=$(echo $MODEL | cut -d'/' -f2)
else
  ORG=""
  BASE_MODEL=$MODEL
fi

FILE_HEADER=$ORG$BASE_MODEL--$DATASET-$SUBSET--$BACKEND-$TEMP-$N_SAMPLES

echo $FILE_HEADER
bigcodebench.generate \
  --tp $NUM_GPU \
  --model $MODEL \
  --bs $BS \
  --temperature $TEMP \
  --n_samples $N_SAMPLES \
  --resume \
  --split $SPLIT \
  --subset $SUBSET \
  --backend $BACKEND \
  --trust_remote_code

bigcodebench.sanitize --samples $FILE_HEADER.jsonl --calibrate

# Check if the ground truth works on your machine
bigcodebench.evaluate --split $SUBSET --samples $FILE_HEADER-sanitized-calibrated.jsonl

# If the execution is slow:
bigcodebench.evaluate --split $SUBSET --samples $FILE_HEADER-sanitized-calibrated.jsonl --parallel 32