'''
Model loader utility for loading and initializing models and tokenizers.
'''

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

class ModelLoader:
    def __init__(self, model_id: str = 'Qwen/Qwen2.5-1.5B', vocab_size=None, **kwargs):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = AutoModelForCausalLM.from_pretrained(model_id).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.vocab_size = len(self.tokenizer) if vocab_size is None else vocab_size
        self.model_kwargs = {}
        self.model_kwargs.update(kwargs)