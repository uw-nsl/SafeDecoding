import gc
import logging
import subprocess
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from fastchat.model import get_conversation_template


def load_model_and_tokenizer(model_path, FP16 = True, tokenizer_path=None, device='cuda:0', **kwargs):
    if FP16:
        model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                trust_remote_code=True,
                **kwargs
            ).to(device).eval()
    else:
        model = AutoModelForCausalLM.from_pretrained(
                model_path,
                trust_remote_code=True,
                **kwargs
            ).to(device).eval()

    if model_path == "timdettmers/guanaco-13b-merged":
        tokenizer_path = "huggyllama/llama-7b"

    tokenizer_path = model_path if tokenizer_path is None else tokenizer_path
    
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        trust_remote_code=True,
        use_fast=False
    )
    
    if 'oasst-sft-6-llama-30b' in tokenizer_path:
        tokenizer.bos_token_id = 1
        tokenizer.unk_token_id = 0
    if 'guanaco' in tokenizer_path:
        tokenizer.eos_token_id = 2
        tokenizer.unk_token_id = 0
    if 'llama-2' in tokenizer_path:
        tokenizer.pad_token = tokenizer.unk_token
        tokenizer.padding_side = 'left'
    if 'falcon' in tokenizer_path:
        tokenizer.padding_side = 'left'
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer


def get_latest_commit_info():
    try:
        # Get the latest commit hash
        commit_hash = subprocess.run(["git", "log", "-1", "--format=%H"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        # Get the latest commit date
        commit_date = subprocess.run(["git", "log", "-1", "--format=%cd"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        # Check if both commands were executed successfully
        if commit_hash.returncode == 0 and commit_date.returncode == 0:
            return commit_hash.stdout.strip(), commit_date.stdout.strip()
        else:
            error_message = commit_hash.stderr if commit_hash.returncode != 0 else commit_date.stderr
            return "Error fetching commit information:", error_message
    except FileNotFoundError:
        # Git not installed or not found in the path
        return "Git is not installed or not found in the path.", ""