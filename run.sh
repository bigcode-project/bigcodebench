BS=5
DATASET=wildcodebench
MODEL=models/gemini-1.5-flash-latest
BACKEND=google
TEMP=0
N_SAMPLES=1
NUM_GPU=1
if [[ $MODEL == *"/"* ]]; then
  ORG=$(echo $MODEL | cut -d'/' -f1)--
  BASE_MODEL=$(echo $MODEL | cut -d'/' -f2)
else
  ORG=""
  BASE_MODEL=$MODEL
fi
echo $ORG$BASE_MODEL--$DATASET--$BACKEND-$TEMP-$N_SAMPLES.jsonl
python -m wildcode.generate \
    --tp $NUM_GPU \
    --model $MODEL \
    --bs $BS \
    --temperature $TEMP \
    --n_samples $N_SAMPLES \
    --resume \
    --dataset $DATASET \
    --backend $BACKEND 

SAMPLE_FILE=$ORG$BASE_MODEL--$DATASET--$BACKEND-$TEMP-$N_SAMPLES.jsonl
python -m wildcode.sanitize --samples $SAMPLE_FILE
python -m wildcode.evaluate --dataset $DATASET --samples $ORG$BASE_MODEL--$DATASET--$BACKEND-$TEMP-$N_SAMPLES-sanitized.jsonl
