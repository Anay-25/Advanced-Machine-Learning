import torch
import torch.nn as nn
import warnings

from jaxtyping import Bool, Float, Int
from transformers import AutoModelForCausalLM
from typing import List

warnings.filterwarnings("ignore")

#References:

# https://arxiv.org/pdf/2401.10774
# https://github.com/FasterDecoding/Medusa


class MedusaTextGenerator:
    def __init__(
        self, 
        model: AutoModelForCausalLM, 
        decoding_strategy: str, 
        eos_id: int, 
        use_no_medusa_heads: int = 5,
        beam_width: int = 2,
        max_output_len: int = 10,
    ) -> None:
        '''
            Initialize the MedusaTextGenerator class.
            
            model: LLM
            decoding_strategy: str describing the decoding strategy to be used.
            eos_id: End-of-sequence token id 
            use_no_medusa_heads: Number of medusa heads to be used (maximum:5) (denoted as S).
            beam_width: Maximum number of candidates that can be present in the beam (denoted as W).
            max_output_len: Maximum number of tokens to be generated.
            
            Do not edit.
        '''
        self.model = model
        self.decoding_strategy = decoding_strategy
        
        self.max_output_len = max_output_len
        self.eos_token_id = eos_id
        self.beam_width = beam_width
        
        assert use_no_medusa_heads <= 5, "The current medusa model supports at max 5 heads"
        self.no_heads = use_no_medusa_heads + 1
        
        if decoding_strategy == "single-head":
            self.generator_func = self.single_head_decoding
        elif decoding_strategy == "multi-head":
            self.generator_func = self.multi_head_decoding
        
    def __call__(
        self, input_ids: Int[torch.Tensor, "batch in_seq_len"]
    ) -> Int[torch.Tensor, "batch out_seq_len"]:
        '''
            Do not edit.
        '''
        return self.generator_func(input_ids)
                
    def single_head_decoding(
        self,
        input_ids: Float[torch.Tensor, "batch in_seq_len"],
    ) -> Int[torch.Tensor, "batch out_seq_len"]:     
        
        generated = []
        current_input = input_ids.clone()
        
        for _ in range(self.max_output_len):
            with torch.no_grad():
                outputs = self.model(current_input)
            
            lm_logits = outputs.logits[:, -1, :]  # shape (1, vocab_size)
            next_token = torch.argmax(lm_logits, dim=-1)
            if next_token.item() == self.eos_token_id:
                break
                
            current_input = torch.cat([
                current_input,
                next_token.unsqueeze(1) 
            ], dim=1)

            generated.append(next_token.item())
        
        return torch.tensor(generated, device=input_ids.device)

    def multi_head_decoding(self, input_ids: torch.Tensor) -> torch.Tensor: 

        generated = []
        assert input_ids.shape[0] == 1  # batch size is 1
        current_input_ids = input_ids.clone()

        for _ in range(self.max_output_len):
            # Forward pass
            with torch.no_grad():
                outputs = self.model(input_ids=current_input_ids)
                all_logits = outputs.logits[:, -1, :]  # logits for last token

            candidates = [current_input_ids.clone()]
            scores = torch.tensor([0.0], device=all_logits.device)

            for head_idx in range(self.no_heads):
                temp_candidates, temp_scores = [], torch.empty(0, device=all_logits.device)

                with torch.no_grad():
                    log_probs = torch.log_softmax(all_logits, dim=-1)  # Recompute per head
                    top_probs, top_tokens = torch.topk(log_probs, self.beam_width, dim=-1)

                for candidate_seq, candidate_score in zip(candidates, scores): #GPT generated
                    # Expand candidate sequences with top-k tokens
                    expanded_sequences = torch.cat(
                        [candidate_seq.repeat(self.beam_width, 1), top_tokens.squeeze(0).unsqueeze(-1)], dim=1
                    )
                    updated_scores = candidate_score + top_probs.squeeze(0)

                    temp_candidates.extend(expanded_sequences)
                    temp_scores = torch.cat([temp_scores, updated_scores])

                # Selecting the best beam_width candidates
                best_indices = torch.topk(temp_scores, self.beam_width).indices
                candidates = [temp_candidates[i] for i in best_indices]
                scores = temp_scores[best_indices]

            # Selecting the best sequence
            best_sequence = candidates[scores.argmax()]
            next_token = best_sequence[:, -1]
            generated.append(next_token.item())

            if next_token.item() == self.eos_token_id:
                break

            current_input_ids = best_sequence  # Updating input sequence

        return torch.tensor(generated, device=input_ids.device)


            