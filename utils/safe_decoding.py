import torch
import numpy as np
import copy
import logging
from peft import PeftModel, PeftModelForCausalLM

class SafeDecoding:
    def __init__(self, model, expert_model, tokenizer, alpha=1, first_m=5, top_k = 10, num_common_tokens = 3, verbose=False):
        self.model = model
        self.expert_model = expert_model
        self.tokenizer = tokenizer
        self.alpha = alpha
        self.first_m = first_m 
        self.top_k = top_k
        self.num_common_tokens = num_common_tokens
        self.verbose = verbose

        logging.info("SafeDecoding initialized.")

    def safedecoding_lora(self, inputs, gen_config=None):
        if gen_config is None:
            gen_config = self.model.generation_config

        max_token_len = gen_config.max_new_tokens
        do_sample = gen_config.do_sample

        # Override the generation config for our decoding
        gen_config.max_new_tokens = 1  # We generate one token at a time
        gen_config.do_sample = False  # We use greedy decoding

        generated_sequence = []
        if self.verbose:
            logging.info(f"Generation config: {gen_config}")

        inputs = {k:v.cuda(self.model.device) for k,v in inputs.items()}
        input_len = inputs['input_ids'].shape[1]

        step = 1  # Keep track of generation steps
        while step <= min(max_token_len, self.first_m):  # Loop until we reach the first m tokens
            # Generate the next token

            output_base = self.model.generate(**inputs,
                                    generation_config=gen_config,
                                    pad_token_id=self.tokenizer.pad_token_id,
                                    return_dict_in_generate=True,
                                    output_scores=True,)

            output_expert = self.expert_model.generate(**inputs,
                                    adapter_names=["expert"],
                                    generation_config=gen_config,
                                    pad_token_id=self.tokenizer.pad_token_id,
                                    return_dict_in_generate=True,
                                    output_scores=True,)
            
            output_base.sequences = output_base.sequences[0].unsqueeze(0)
            output_base.scores = output_base.scores[0][0].unsqueeze(0)
            output_expert.sequences = output_expert.sequences[0].unsqueeze(0)
            output_expert.scores = output_expert.scores[0][0].unsqueeze(0)

            # Process the scores to get the top tokens
            k = self.top_k  # Change this to display more or less tokens
            scores_base = output_base.scores[-1].squeeze()  # Get the scores of the last token
            scores_base = torch.nn.functional.log_softmax(scores_base, dim=-1)
            topk_scores_base, topk_indices_base = scores_base.topk(k) 
            
            scores_expert = output_expert.scores[-1].squeeze()  # Get the scores of the last token
            scores_expert = torch.nn.functional.log_softmax(scores_expert, dim=-1)
            topk_scores_expert, topk_indices_expert = scores_expert.topk(k) 

            sorted_indices_base = torch.argsort(scores_base, descending=True)
            sorted_indices_expert = torch.argsort(scores_expert, descending=True)

            # Step 1: Define Sample Space
            common_tokens = set()
            iter_range = self.num_common_tokens
            while len(common_tokens) < self.num_common_tokens:
                current_indices_base = sorted_indices_base[:iter_range]
                current_indices_expert = sorted_indices_expert[:iter_range]

                common_in_iteration = set(current_indices_base.tolist()) & set(current_indices_expert.tolist())
                common_tokens.update(common_in_iteration)

                iter_range += 1

                if iter_range > min(len(sorted_indices_base), len(sorted_indices_expert)):
                    break

            # Display the top tokens
            if self.verbose and step == 1:
                logging.info("\n-----------------------------------------------")
                logging.info(f"Generation Step {step}")
                logging.info("Original Model")
                logging.info("|No. | Token ID | Token   | Log Prob | Prob    |")
                logging.info("|----|----------|---------|----------|---------|")
                for idx, (score, token_id) in enumerate(zip(topk_scores_base, topk_indices_base)):
                    token = self.tokenizer.decode(token_id.item())
                    prob = torch.exp(score)
                    logging.info(f"{idx+1:4d} | {token_id:8d} | {token:7s} | {score:.3f}    | {prob:.2%} |")

                logging.info("Expert Model")
                logging.info("|No. | Token ID | Token   | Log Prob | Prob    |")
                logging.info("|----|----------|---------|----------|---------|")
                for idx, (score, token_id) in enumerate(zip(topk_scores_expert, topk_indices_expert)):
                    token = self.tokenizer.decode(token_id.item())
                    prob = torch.exp(score)
                    logging.info(f"{idx+1:4d} | {token_id:8d} | {token:7s} | {score:.3f}    | {prob:.2%} |")

            intersection_indices = torch.tensor(list(common_tokens), device=self.model.device)
            
            # Step 2: New Probability Calculation
            updated_scores = []
            for token_id in intersection_indices:
                # Steer scores
                # new_score = (1-self.alpha) * scores_base[token_id] + self.alpha * scores_expert[token_id]
                # updated_scores.append(new_score)

                # Steer probabilities
                prob_diff = torch.exp(scores_expert[token_id]) - torch.exp(scores_base[token_id])
                updated_prob = torch.exp(scores_base[token_id]) + self.alpha * prob_diff
                # Floor the probability to 1e-8 to avoid log(0)
                updated_prob = updated_prob if updated_prob > 0 else torch.tensor(1e-8, device=self.model.device)
                updated_score = torch.log(updated_prob)
                updated_scores.append(updated_score)

                if self.verbose:
                    logging.info(f"----------------token id: {token_id}-----------------")
                    logging.info(f"Prob Base: {torch.exp(scores_base[token_id])}")
                    logging.info(f"Prob Expert: {torch.exp(scores_expert[token_id])}")
                    logging.info(f"Base score: {scores_base[token_id]}")
                    logging.info(f"Expert score: {scores_expert[token_id]}")
                    logging.info(f"Updated Probability: {updated_prob}")
                    logging.info(f"Updated Score: {updated_score}")

            # Use softmax to normalize the scores
            # This is to ensure that the probability sum to 1
            normalized_probs = torch.nn.functional.softmax(torch.tensor(updated_scores).float(), dim=0)

            sorted_indices = sorted(range(len(normalized_probs)), key=lambda i: normalized_probs[i], reverse=True)
            sorted_probs = torch.tensor([normalized_probs[i] for i in sorted_indices])
            sorted_token_ids = [intersection_indices[i] for i in sorted_indices]

            if self.verbose:
                logging.info("\n-----------------------------------------------")
                logging.info(f"Generation Step {step}")
                logging.info("|No. | Token ID | Token   | Log Prob | Prob    |")
                logging.info("|----|----------|---------|----------|---------|")
                for idx, (prob, token_id) in enumerate(zip(sorted_probs, sorted_token_ids)):
                    token = self.tokenizer.decode(token_id.item())
                    score = torch.log(prob)
                    logging.info(f"{idx+1:4d} | {token_id:8d} | {token:7s} | {score:.3f}    | {prob:.2%} |")

            ### Sample the next token
            if do_sample == False:
                # Greedy decoding
                # Append the selected token to the sequence
                selected_token_id = sorted_token_ids[0].unsqueeze(0)
            elif gen_config.top_p != None and do_sample == True:
                # Top-p sampling, sample from the top-p tokens
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                p_index = torch.where(cumulative_probs >= gen_config.top_p)[0][0]
                sorted_top_p_token_ids = sorted_token_ids[:p_index + 1]
                sorted_top_p_probs = sorted_probs[:p_index + 1]
                sorted_top_p_scores = torch.log(sorted_top_p_probs)
                if self.verbose:
                    logging.info(f"Top-p token ids: {sorted_top_p_token_ids}")
                    logging.info(f"Top-p scores: {sorted_top_p_scores}")
                    logging.info(f"Top-p probabilities: {sorted_top_p_probs}")
                
                # Sample from the top-p tokens
                selected_token_id = sorted_top_p_token_ids[torch.multinomial(torch.softmax(sorted_top_p_scores, dim=-1), 1)].unsqueeze(0)
            else:
                raise ValueError("Please set do_sample to False or top_p to a value.")

            if self.verbose:
                logging.info(f"Selected token: {self.tokenizer.decode(selected_token_id.item())}, ID: {selected_token_id.item()}")
            generated_sequence.append(selected_token_id.item())

            # if the chosen token id is eos, then stop
            if selected_token_id.item() == self.tokenizer.eos_token_id:
                break

            inputs['input_ids'] = torch.cat([inputs['input_ids'], selected_token_id.unsqueeze(0)], dim=1)
            inputs['attention_mask'] = torch.cat([inputs['attention_mask'], torch.tensor([[1]], device=self.model.device)], dim=1)

            step += 1

            # Free up memory
            del output_base, output_expert


        # Use the normal model to generate the rest of the tokens
        # Early stop if the last token is eos
        if generated_sequence[-1] == self.tokenizer.eos_token_id:
            logging.info("Early stop triggered.")
        else:
            remaining_steps = max_token_len - min(max_token_len, self.first_m)
            gen_config.max_new_tokens = remaining_steps
            gen_config.do_sample = do_sample
            output_base = self.model.generate(**inputs,
                                    generation_config=gen_config,
                                    pad_token_id=self.tokenizer.pad_token_id,
                                    return_dict_in_generate=True,
                                    output_scores=True,)
            
            generated_sequence = output_base.sequences[0].tolist()[input_len:]

        # logging.info generated sequence
        logging.info(f"Generated sequence: {self.tokenizer.decode(generated_sequence)}")

        return self.tokenizer.decode(generated_sequence), len(generated_sequence)
    
    
    def generate_baseline(self, inputs, gen_config=None):
        if gen_config is None:
            gen_config = self.model.generation_config
        
        if self.verbose:
            logging.info(f"Generation config: {gen_config}")

        inputs = {k:v.cuda(self.model.device) for k,v in inputs.items()}

        output_base = self.model.generate(**inputs,
                            generation_config=gen_config,
                            pad_token_id=self.tokenizer.pad_token_id,
                            return_dict_in_generate=True,
                            output_scores=True,)
        
        generated_sequence = output_base.sequences[0][inputs["input_ids"].shape[1]:]
        logging.info(f"Generated sequence: {self.tokenizer.decode(generated_sequence)}")
        
        return self.tokenizer.decode(generated_sequence), len(generated_sequence)