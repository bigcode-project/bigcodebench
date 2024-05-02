#codeqwen1.5-7b-chat
#gpt-3.5-turbo-0125
MODEL=codeqwen1.5-7b-chat
BS=1
TEMP=0
N_SAMPLES=1
DATASET=openeval

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
# python codegen/inspection.py --dataset openeval --eval-results $MODEL-$DATASET-$TEMP-$N_SAMPLES-sanitized_eval_results.json