import argparse
import os
import json
import openai 
import random
from pathlib import Path
from itertools import combinations
from string import Template
from tqdm import tqdm
from threading import get_ident
from concurrent.futures import ThreadPoolExecutor
from .utils import (
    better_json_loads,
    openai_chat_request, 
    MULTI_SCORE_TEMPLATE,
    SAFETY_SCORE_TEMPLATE, 
)
import numpy as np 
 
 
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--report_only', action='store_true')
    
    parser.add_argument("--mode", type=str, default="pairwise", required=True)
    parser.add_argument("--first_file", type=str, required=False)
    parser.add_argument("--second_file", type=str, required=False)
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--end_idx", type=int, default=-1) 
    parser.add_argument("--reference_file", type=str, required=False) 
    parser.add_argument("--save_interval", type=int, default=3)
    
    # Prompt configs 
    parser.add_argument("--max_words_to_eval", type=int, default=-1)
    
    # OpenAI Configs
    parser.add_argument("--api_key", type=str, default=None)
    parser.add_argument("--model", type=str, default="gpt-4-0314")
    parser.add_argument("--engine", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_tokens", type=int, default=1024)
    
    args = parser.parse_args() 
    if args.api_key is not None:
        openai.api_key = args.api_key 
    
    if args.report_only:
        print("\nloading:", args.output_file)
        assert os.path.exists(args.output_file) 
        
    return args

def report(results, mode, args):
    
    if mode.startswith("score"):
        cnt = 0
        lens_cand = []
        if "_multi" not in mode and "_safety" not in mode:
            scores = []
        else:
            scores = {} 
        eval_res = {}
        
        for item in results:
            # avoid non type error
            if "parsed_result" in item and item["parsed_result"] is not None:
                d = item
                d["len_cand"] = len(d["output_cand"].split())
                lens_cand.append(item["len_cand"])
                if "_multi" not in mode and "_safety" not in mode:
                    score = item["parsed_result"]["score"]
                    scores.append(float(score))
                else:
                    for aspect, result in item["parsed_result"].items():
                        if aspect not in scores:
                            scores[aspect] = []
                        if result["score"] == "N/A":
                            result["score"] = 5.0
                        scores[aspect].append(float(result["score"]))
                
                cnt += 1
        if "_multi" not in mode and "_safety" not in mode:
            eval_res = {"total": cnt, "average_score": float(np.mean(scores)), "std": float(np.std(scores))}
        else:
            eval_res = {"total": cnt}
            for aspect, score_list in scores.items():
                eval_res[aspect + "_mean"] = float(np.mean(score_list))
                # eval_res[aspect + "_std"] = float(np.std(score_list))
                
        eval_res["avg_lens"] = float(np.mean(lens_cand))
    
    eval_res["output_file"] = args.output_file
    return eval_res
                
def gpt_eval(results, args):
    # try to load the existing results from args.output_file 
    if os.path.exists(args.output_file):
        cnt = 0 
        with open(args.output_file, "r") as f:
            existing_results = json.load(f) 
        for i in range(len(existing_results)):
            e = existing_results[i]
            t = results[i+args.start_idx]
            if e["prompt"] != t["prompt"]:
                continue
            # if e["prompt"] == t["prompt"] and e["result"] != "N/A":
            #     results[i]["result"] = e["result"]
            #     cnt += 1 
            if "result" in e:
                t["result"] = e["result"]
                if "parsed_result" in e: 
                    t["parsed_result"] = e["parsed_result"]
                cnt += 1
        print(f"loading {cnt} results from {args.output_file}")
    openai_args = {
        "prompt": "TODO",
        "api_key": args.api_key,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "stop": []
    }
    if args.model:
        openai_args['model'] = args.model
    if args.engine:
        openai_args['engine'] = args.engine
        
    def api(ind, item, **kwargs):
        result = openai_chat_request(**kwargs)
        result = result[0]
        if args.mode == "tag":
            return result 
        
        result = result.replace("```", "")
        if '\\\\"' in result:
            result = result.replace('\\\\"', '\\"')
        else:
            result = result.replace("\\", "\\\\")
        result = result.strip()
        if result[0] != "{" or result[0] != "}":
            start_index = result.find("{")
            end_index = result.rfind("}") + 1
            result = result[start_index:end_index]
        
        
        try:
            # json.loads(result)
            better_json_loads(result)
        except Exception as e:
            print(ind)
            print(e)
            print(result)
            raise e
        return result
    
    results = results[args.start_idx:args.end_idx] # for debug
    for ind, item in tqdm(enumerate(results), total=len(results), desc=f"Evaluating: {args.output_file} "):
        if item["result"] != "N/A":
            if args.mode != "tag":
                results[ind]["parsed_result"] = better_json_loads(results[ind]["result"])
            print(f"Skipping {ind} for {args.output_file}")
            continue
        # print(f"\nNot Skipping {ind}") 
    
        openai_args["prompt"] = item["prompt"]
        try:
            result = api(ind, item, **openai_args)
            results[ind]["result"] = result
            if args.mode != "tag":
                results[ind]["parsed_result"] = better_json_loads(results[ind]["result"])
            else:
                results[ind]["parsed_result"] = "N/A"
        except Exception as e:
            print(e)
        
        # print("Done!") 
        if ind % args.save_interval == 0 or ind == len(results)-1:
            with open(args.output_file, "w") as f:
                json.dump(results, f, indent=2) 
    with open(args.output_file, "w") as f:
        json.dump(results, f, indent=2)
    return results 

def shorten(text, K=-1):
    if K > 0 and len(text.split(" ")) > K:
        text = " ".join(text.split(" ")[:K]) + "... (truncated)"
    return text
 
    
def score_eval(args):
    results = []
    with open(args.first_file, 'r') as f:
        candidates = json.load(f) 
    references = [None] * len(candidates)
        
    L = min(len(candidates), len(references))
    if args.end_idx < 0:
        args.end_idx = L
    print(f"# examples in candidates: {len(candidates)}; # examples in references: {len(references)}; We take {args.end_idx-args.start_idx} for evaluation.")
    candidates = candidates[:L]
    references = references[:L]
    
    results = []
    for itemA, itemB in zip(candidates, references):
        instruction = itemA["instruction"] 
        if args.mode == "score_multi":
            A = itemA["output"]
            A = shorten(A)
            prompt = Template(MULTI_SCORE_TEMPLATE).substitute(
                instruction = instruction, 
                candidate = A
            )
        elif args.mode == "score_safety":
            A = itemA["output"]
            A = shorten(A)
            prompt = Template(SAFETY_SCORE_TEMPLATE).substitute(
                instruction = instruction, 
                candidate = A
            )
        else:
            prompt = "N/A" # for reward-based eval 
        d = {}
        d["id"] = itemA.get("id", len(results))
        d["input"] = instruction
        d["output_cand"] = itemA["output"]
        d["generator_cand"] = itemA["generator"]  
        d["eval_config"] = {"mode": args.mode, "gpt": args.model, "max_words": args.max_words_to_eval}
        d["prompt"] = prompt
        d["result"] = "N/A" 
        results.append(d)
    return results 
 
  

def main():
    random.seed(42)
    args = get_args()
    
    if args.report_only:
        with open(args.output_file) as f:
            results = json.load(f)
        if args.end_idx > 0:
            results = results[:args.end_idx]
        eval_res = report(results, args.mode, args)
        print(json.dumps(eval_res, indent=2))
        with open(args.output_file.replace(".json",".eval_res.json"), "w") as f:
            json.dump(eval_res, f, indent=2)
            print("Evaluation results saved to:", f.name)
        exit()
    
    if args.mode.startswith("score"):
        results = score_eval(args)
        results = gpt_eval(results, args) 
    else:
        print("Not implemented yet!")

if __name__ == "__main__": 
    main()
     