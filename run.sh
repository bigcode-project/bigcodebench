BS=5
DATASET=wildcodebench
MODEL=gpt-3.5-turbo-0125
BACKEND=openai
TEMP=0
N_SAMPLES=1
NUM_GPU=1
NL2CODE=True
if [[ $MODEL == *"/"* ]]; then
  ORG=$(echo $MODEL | cut -d'/' -f1)--
  BASE_MODEL=$(echo $MODEL | cut -d'/' -f2)
else
  ORG=""
  BASE_MODEL=$MODEL
fi

if [ "$NL2CODE" = "True" ]; then
  FILE_HEADER=$ORG$BASE_MODEL--$DATASET-nl2c--$BACKEND-$TEMP-$N_SAMPLES
else
  FILE_HEADER=$ORG$BASE_MODEL--$DATASET-c2c--$BACKEND-$TEMP-$N_SAMPLES
fi

echo $FILE_HEADER
wildcode.generate \
  --tp $NUM_GPU \
  --model $MODEL \
  --bs $BS \
  --temperature $TEMP \
  --n_samples $N_SAMPLES \
  --resume \
  --dataset $DATASET \
  --nl2code $NL2CODE \
  --backend $BACKEND 

wildcode.sanitize --samples $FILE_HEADER.jsonl
wildcode.evaluate --dataset $DATASET --samples $FILE_HEADER-sanitized.jsonl