#codeqwen1.5-7b-chat
#gpt-3.5-turbo-0125
#deepseek-coder-33b-instruct
#starcoder2-15b-instruct-v0.1
BS=10
MODEL=starcoder2-15b-instruct-v0.1
DATASET=openeval
TEMPS=(0.2) # print("Starting process!!!")
N_SAMPLESES=(10) # 50
for TEMP in ${TEMPS[@]}; do
  for N_SAMPLES in ${N_SAMPLESES[@]}; do
    rm -rf $MODEL-$DATASET-$TEMP-$N_SAMPLES.jsonl
    python codegen/generate.py \
        --model $MODEL \
        --bs $BS \
        --temperature $TEMP \
        --n_samples $N_SAMPLES \
        --resume \
        --dataset $DATASET \
        --save-path $MODEL-$DATASET-$TEMP-$N_SAMPLES.jsonl
    rm -rf $MODEL-$DATASET-$TEMP-$N_SAMPLES-sanitized.jsonl
    openeval.sanitize --samples $MODEL-$DATASET-$TEMP-$N_SAMPLES.jsonl
    rm -rf $MODEL-$DATASET-$TEMP-$N_SAMPLES-sanitized_eval_results.json
    openeval.evaluate --dataset openeval --samples $MODEL-$DATASET-$TEMP-$N_SAMPLES-sanitized.jsonl
    # python codegen/inspection.py --dataset openeval --eval-results $MODEL-$DATASET-$TEMP-$N_SAMPLES-sanitized_eval_results.json --in-place
  done
done
