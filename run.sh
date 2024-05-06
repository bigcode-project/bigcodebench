#Qwen/CodeQwen1.5-7B-Chat
#gpt-3.5-turbo-0125
#deepseek-coder-33b-instruct
#bigcode/starcoder2-15b-instruct-v0.1
#bigcode/starcoder2-15b
#gpt-4-turbo-2024-04-09
BS=10
MODEL=gpt-3.5-turbo-0125
DATASET=wildcodebench
BACKEND=openai
TEMPS=(0)
N_SAMPLESES=(1)

for TEMP in ${TEMPS[@]}; do
  for N_SAMPLES in ${N_SAMPLESES[@]}; do
    rm -rf $MODEL-$DATASET-$TEMP-$N_SAMPLES.jsonl
    wildcode.generate \
        --model $MODEL \
        --bs $BS \
        --temperature $TEMP \
        --n_samples $N_SAMPLES \
        --resume \
        --dataset $DATASET \
        --backend $BACKEND

    # check if "/" in the model name. I
    if [[ $MODEL == *"/"* ]]; then
      # split the model name by "/", first part is the organization name, second part is the model name
      ORG=$(echo $MODEL | cut -d'/' -f1)
      MODEL=$(echo $MODEL | cut -d'/' -f2)
      MODEL=$ORG--$MODEL
    fi

    SAMPLE_FILE=$MODEL--$DATASET--$BACKEND-$TEMP-$N_SAMPLES.jsonl
    rm -rf $MODEL--$DATASET--$BACKEND-$TEMP-$N_SAMPLES-sanitized.jsonl
    wildcode.sanitize --samples $SAMPLE_FILE
    rm -rf $MODEL--$DATASET--$BACKEND-$TEMP-$N_SAMPLES-sanitized_eval_results.json
    wildcode.evaluate --dataset wildcode --samples $MODEL--$DATASET--$BACKEND-$TEMP-$N_SAMPLES-sanitized.jsonl
    # python codegen/inspection.py --dataset wildcode --eval-results $MODEL-$DATASET-$TEMP-$N_SAMPLES-sanitized_eval_results.json --in-place
  done
done
