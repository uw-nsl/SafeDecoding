import torch
import numpy as np

def generate(model, tokenizer, input_ids, gen_config=None):
    if gen_config is None:
        gen_config = model.generation_config
        gen_config.max_new_tokens = 32
        
    input_ids = input_ids.to(model.device).unsqueeze(0)
    input_len = input_ids.shape[1]
    attn_masks = torch.ones_like(input_ids).to(model.device)
    output_ids = model.generate(input_ids=input_ids, 
                                attention_mask=attn_masks, 
                                generation_config=gen_config,
                                pad_token_id=tokenizer.pad_token_id)[0]

    # print(f"decoded output: [{tokenizer.decode(output_ids)}]")

    return output_ids[input_len:]