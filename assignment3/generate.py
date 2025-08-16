import torch
import torch.nn as nn
import warnings

from jaxtyping import Bool, Float, Int
from transformers import AutoModelForCausalLM
from typing import List

warnings.filterwarnings("ignore")

#References

# https://www.assemblyai.com/blog/decoding-strategies-how-llms-choose-the-next-word
# https://huggingface.co/blog/mlabonne/decoding-strategies
# https://pytorch.org/docs/stable/distributions.html


class TextGenerator:
    def __init__(
        self, 
        model: AutoModelForCausalLM, 
        decoding_strategy: str, 
        eos_id: int, 
        max_output_len: int = 10,
        tau: int = 0.5,
        k: int = 8,
        p: int = 0.5
    ) -> None:
        '''
            Initialize the TextGenerator class.
            
            model: LLM
            decoding_strategy: str describing the decoding strategy to be used.
            eos_id: End-of-sequence token id 
            max_output_len: Maximum number of tokens to be generated.
            tau: Temperature parameter for random sampling
            k: Top-k parameter for top-k sampling
            p: Cumulative probability threshold for nucleus sampling
            
            Do not edit.
        '''
        self.model = model
        self.decoding_strategy = decoding_strategy
        
        self.max_output_len = max_output_len
        self.eos_token_id = eos_id
        self.tau = tau
        self.k = k 
        self.p = p
        
        if decoding_strategy == "greedy":
            self.generator_func = self.greedy_decoding
        elif decoding_strategy == "random":
            self.generator_func = self.random_sampling
        elif decoding_strategy == "topk":
            self.generator_func = self.topk_sampling
        elif decoding_strategy == "nucleus":
            self.generator_func = self.nucleus_sampling

    def __call__(
        self, 
        input_ids: Int[torch.Tensor, "batch in_seq_len"], 
    ) -> Int[torch.Tensor, "batch out_seq_len"]:
        '''
            Do not edit.
        '''
        return self.generator_func(input_ids)
                
    def greedy_decoding(
        self,
        input_ids: Int[torch.Tensor, "batch in_seq_len"],
    ) -> Int[torch.Tensor, "batch out_seq_len"]: 
        
        generated = []
        current_input = input_ids.clone()

        for _ in range(self.max_output_len):
            # Forward pass through the model
            with torch.no_grad():
                outputs = self.model(current_input)
                
            # getting logits
            logits = outputs.logits[:, -1, :]  # shape (1, vocab_size)

            # selecting token with highest probability
            next_token = torch.argmax(logits, dim=-1)  # shape (1,)
            
            # Append generated token first
            generated.append(next_token.item())
            
            # Check for EOS token
            if next_token == self.eos_token_id:
                break
                
            # Update input with proper dimensionality, GPT generated
            current_input = torch.cat([
                current_input, 
                next_token.unsqueeze(1)  # Add sequence dimension (1, 1)
            ], dim=1)

        return torch.tensor(generated, device=input_ids.device)

    def random_sampling(
        self, 
        input_ids: Int[torch.Tensor, "batch in_seq_len"]
    ) -> Int[torch.Tensor, "batch out_seq_len"]:
        
        generated = []
        current_input = input_ids.clone()
        
        for _ in range(self.max_output_len):
            with torch.no_grad():
                outputs = self.model(current_input)
        
            logits = outputs.logits[:, -1, :]
            probs = torch.softmax(logits, dim=-1)
            
            # temperature scaling
            scaled_probs = probs ** (1.0 / self.tau)
            
            # Numerical stability: avoid division by zero, GPT generated
            scaled_probs_sum = scaled_probs.sum(dim=-1, keepdim=True)
            scaled_probs_sum = torch.clamp(scaled_probs_sum, min=1e-10)  # prevent NaN
            scaled_probs = scaled_probs / scaled_probs_sum
            
            # Sample from modified distribution
            next_token = torch.multinomial(
                scaled_probs.squeeze(0),
                num_samples=1
            ) 
        
            generated.append(next_token.item())
            if next_token == self.eos_token_id:
                break
    
            current_input = torch.cat([
                current_input, 
                next_token.unsqueeze(1)
            ], dim=1)
        
        return torch.tensor(generated, device=input_ids.device)
        
        
        
    def topk_sampling(
        self, 
        input_ids: Int[torch.Tensor, "batch in_seq_len"]
    ) -> Int[torch.Tensor, "batch out_seq_len"]:
        
        generated = []
        current_input = input_ids.clone()
    
        for _ in range(self.max_output_len):
            with torch.no_grad():
                outputs = self.model(current_input)

            logits = outputs.logits[:, -1, :]
            probs = torch.softmax(logits, dim=-1)

            # selecting top-k tokens
            topk_probs, topk_indices = torch.topk(probs, self.k , dim=-1)
            
            # Creating mask, GPT generated
            mask = torch.zeros_like(probs, dtype=torch.bool)
            mask.scatter_(-1, topk_indices, True)
            
            # zeroing out
            probs_topk = probs * mask
            probs_topk /= probs_topk.sum(dim=-1, keepdim=True)  # normalize\

            next_token = torch.multinomial(
                probs_topk.squeeze(0), 
                num_samples=1
            )  

            if next_token.item() == self.eos_token_id:
                break

            current_input = torch.cat([
                current_input,
                next_token.unsqueeze(1)
            ], dim=1)
            
            generated.append(next_token.item())

        return torch.tensor(generated, device=input_ids.device)
    
    def nucleus_sampling(
        self, 
        input_ids: Int[torch.Tensor, "batch in_seq_len"]
    ) -> Int[torch.Tensor, "batch out_seq_len"]:
        generated = []
        current_input = input_ids.clone()
        
        for _ in range(self.max_output_len):
            with torch.no_grad():
                outputs = self.model(current_input)

            logits = outputs.logits[:, -1, :]
            probs = torch.softmax(logits, dim=-1) 

            # sorting in descending
            sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
            
            # cumulative oprob
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
            
            # cutting at specific threshold
            cutoff = (cumulative_probs < self.p).sum(dim=-1) + 1
            cutoff = torch.clamp(cutoff, max=probs.size(-1))
            
            # indices of tokens, GPT generated
            nucleus_indices = sorted_indices[..., :cutoff]
            
            # masking
            mask = torch.zeros_like(probs, dtype=torch.bool)
            mask.scatter_(-1, nucleus_indices, True)
    
            probs_nucleus = probs * mask
            probs_nucleus /= probs_nucleus.sum(dim=-1, keepdim=True)

           
            next_token = torch.multinomial(
                probs_nucleus.squeeze(0), 
                num_samples=1
            )  

            if next_token.item() == self.eos_token_id:
                break
                
            current_input = torch.cat([
                current_input,
                next_token.unsqueeze(1)
            ], dim=1)
            
            generated.append(next_token.item())

        return torch.tensor(generated, device=input_ids.device)