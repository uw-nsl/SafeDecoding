## MT-Bench

#### Step 1. Generate model answers to MT-bench questions
```
python gen_model_answer.py --model-path [MODEL-PATH] --model-id [MODEL-ID] --defense [DEFENDER]
```
Arguments:
  - `[MODEL-PATH]` is the path to the weights, which can be a local folder or a Hugging Face repo ID.
  - `[MODEL-ID]` is a name you give to the model.
  - `[DEFENDER]` is the defender's name, e.g., SafeDecoding

e.g.,
```
python gen_model_answer.py --model-path lmsys/vicuna-7b-v1.5 --model-id vicuna-7b-v1.5-safedecoding --defense SafeDecoding
```
The answers will be saved to `data/mt_bench/model_answer/[MODEL-ID].jsonl`.

#### Step 2. Generate GPT-4 judgments
There are several options to use GPT-4 as a judge, such as pairwise winrate and single-answer grading.
In MT-bench, we recommend single-answer grading as the default mode.
This mode asks GPT-4 to grade and give a score to model's answer directly without pairwise comparison.
For each turn, GPT-4 will give a score on a scale of 10. We then compute the average score on all turns.

Note that you need to **create a new environment** for generating judgments, as MT-bench only supports `openai==0.28.1` (sad)

```
export OPENAI_API_KEY=XXXXXX  # set the OpenAI API key
python gen_judgment.py --model-list [LIST-OF-MODEL-ID] --parallel [num-concurrent-api-call]
```

e.g.,
```
python gen_judgment.py --model-list vicuna-13b-v1.3 alpaca-13b llama-13b claude-v1 gpt-3.5-turbo gpt-4 --parallel 10
```
The judgments will be saved to `data/mt_bench/model_judgment/gpt-4_single.jsonl`

#### Step 3. Show MT-bench scores

- Show the scores for selected models
  ```
  python show_result.py --model-list vicuna-13b-v1.3 alpaca-13b llama-13b claude-v1 gpt-3.5-turbo gpt-4
  ```
- Show all scores
  ```
  python show_result.py
  ```

---
