import torch
import torch.nn as nn
import warnings

from jaxtyping import Bool, Float, Int
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List

warnings.filterwarnings("ignore")

#References

# https://wangyy395.medium.com/implement-a-trie-in-python-e8dd5c5fde3a
# https://huggingface.co/blog/vivien/llm-decoding-with-regex-constraints
# https://medium.com/thedeephub/all-you-need-to-know-about-tokenization-in-llms-7a801302cf54


class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end = False

class Trie:
    def __init__(self):
        self.root = TrieNode()
    
    def insert(self, token_ids: List[int]):
        node = self.root
        for token in token_ids:
            if token not in node.children:
                node.children[token] = TrieNode()
            node = node.children[token]
        node.is_end = True


class ConstrainedTextGenerator:
    def __init__(
        self, 
        model: AutoModelForCausalLM, 
        tokenizer: AutoTokenizer, 
        eos_id: int, 
        max_output_len: int = 10,
    ) -> None:
        '''
            Initialize the ConstrainedTextGenerator class.
            
            model: LLM
            tokenizer: LLM's tokenizer.
            eos_id: End-of-sequence token id 
            max_output_len: Maximum number of tokens to be generated.
            
            Do not edit.
        '''
        self.model = model
        
        self.max_output_len = max_output_len
        self.eos_token_id = eos_id
        
        self.tokenizer = tokenizer

    def __call__(
        self, input_ids: Int[torch.Tensor, "batch in_seq_len"], word_list: list
    ) -> Int[torch.Tensor, "batch out_seq_len"]:
        
        trie = Trie()
        for word in word_list:
            tokens = self.tokenizer.tokenize(word)
            token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
            trie.insert(token_ids)
        
        generated = []
        current_input = input_ids.clone()
        current_node = trie.root
        
        for _ in range(self.max_output_len):
            with torch.no_grad():
                outputs = self.model(current_input)
            
            logits = outputs.logits[:, -1, :] 
            
            allowed_tokens = []
            if current_node.is_end:
                # Can start new word or EOS
                allowed_tokens = list(trie.root.children.keys()) + [self.eos_token_id]
            else:
                # Continue current word
                allowed_tokens = list(current_node.children.keys())

            #Masking
            mask = torch.ones_like(logits, dtype=torch.bool)
            if allowed_tokens:
                mask[:, allowed_tokens] = False
            logits[mask] = -float('inf')
            
            # greedy
            next_token = torch.argmax(logits, dim=-1)
            
            # Handle EOS
            if next_token.item() == self.eos_token_id:
                break
                
            # Update trie state, GPT generated
            if current_node.is_end:
                current_node = trie.root.children.get(next_token.item(), trie.root)
            else:
                current_node = current_node.children.get(next_token.item(), trie.root)

            current_input = torch.cat([
                current_input,
                next_token.unsqueeze(1)
            ], dim=1)
            
            generated.append(next_token.item())
        
        return torch.tensor(generated, device=input_ids.device)
        
        