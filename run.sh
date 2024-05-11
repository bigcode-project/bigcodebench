#Qwen/CodeQwen1.5-7B-Chat
#gpt-3.5-turbo-0125
#deepseek-ai/deepseek-coder-33b-instruct
#bigcode/starcoder2-15b-instruct-v0.1
#bigcode/starcoder2-15b
#gpt-4-turbo-2024-04-09
BS=5
DATASET=wildcodebench
# MODELS=(Qwen/CodeQwen1.5-7B-Chat)
# BACKEND=vllm
MODELS=(gpt-4-turbo-2024-04-09)
BACKEND=openai
TEMPS=(0)
N_SAMPLESES=(1)

for MODEL in ${MODELS[@]}; do
  for TEMP in ${TEMPS[@]}; do
    for N_SAMPLES in ${N_SAMPLESES[@]}; do
      python -m wildcode.generate \
          --tp 1 \
          --model $MODEL \
          --bs $BS \
          --temperature $TEMP \
          --n_samples $N_SAMPLES \
          --resume \
          --dataset $DATASET \
          --backend $BACKEND

      if [[ $MODEL == *"/"* ]]; then
        ORG=$(echo $MODEL | cut -d'/' -f1)
        MODEL=$(echo $MODEL | cut -d'/' -f2)
        MODEL=$ORG--$MODEL
      fi

      SAMPLE_FILE=$MODEL--$DATASET--$BACKEND-$TEMP-$N_SAMPLES.jsonl
      python -m wildcode.sanitize --samples $SAMPLE_FILE
      rm -rf $MODEL--$DATASET--$BACKEND-$TEMP-$N_SAMPLES-sanitized_eval_results.json
      python -m wildcode.evaluate --dataset $DATASET --samples $MODEL--$DATASET--$BACKEND-$TEMP-$N_SAMPLES-sanitized.jsonl
      # python -m wildcode.inspect --dataset $DATASET --eval-results $MODEL--$DATASET--$BACKEND-$TEMP-$N_SAMPLES-sanitized_eval_results.json --in-place
    done
  done
done
