BS=5
DATASET=bigcodebench
MODEL=gpt-3.5-turbo-0125
BACKEND=openai
TEMP=0
N_SAMPLES=1
NUM_GPU=1
SUBSET=instruct
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
  --id_range 0 1 \
  --tp $NUM_GPU \
  --model $MODEL \
  --bs $BS \
  --temperature $TEMP \
  --n_samples $N_SAMPLES \
  --resume \
  --subset $SUBSET \
  --backend $BACKEND

bigcodebench.sanitize --samples $FILE_HEADER.jsonl
bigcodebench.evaluate --subset $SUBSET --samples $FILE_HEADER-sanitized.jsonl