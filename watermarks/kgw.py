'''
Implementation of the KGW watermarking algorithm.
Reference:https://arxiv.org/abs/2301.10226 and https://github.com/THU-BPM/MarkLLM
'''


import torch
from math import sqrt
from functools import partial

from transformers import LogitsProcessor, LogitsProcessorList

from watermarks.base import BaseWatermark
from configs.base import WatermarkConfig
from data_structure.base import WatermarkInput, WatermarkOutput, DetectResult

class KGWUtils:
    """Utility class for KGW algorithm, contains helper functions."""

    def __init__(self, config: WatermarkConfig, **kwargs) -> None:
        self.config = config
        self.rng = torch.Generator(device=self.config.device)
        self.rng.manual_seed(self.config.hash_key)
        self.prf = torch.randperm(self.config.vocab_size, device=self.config.device, generator=self.rng)
        self.f_scheme_map = {"time": self._f_time, "additive": self._f_additive, "skip": self._f_skip, "min": self._f_min}
        self.window_scheme_map = {"left": self._get_greenlist_ids_left, "self": self._get_greenlist_ids_self}

    def _f(self, input_ids: torch.LongTensor) -> int:
        """Get the previous token."""
        return int(self.f_scheme_map[self.config.f_scheme](input_ids))
    
    def _f_time(self, input_ids: torch.LongTensor):
        """Get the previous token time."""
        time_result = 1
        for i in range(0, self.config.prefix_length):
            time_result *= input_ids[-1 - i].item()
        return self.prf[time_result % self.config.vocab_size]
    
    def _f_additive(self, input_ids: torch.LongTensor):
        """Get the previous token additive."""
        additive_result = 0
        for i in range(0, self.config.prefix_length):
            additive_result += input_ids[-1 - i].item()
        return self.prf[additive_result % self.config.vocab_size]
    
    def _f_skip(self, input_ids: torch.LongTensor):
        """Get the previous token skip."""
        return self.prf[input_ids[- self.config.prefix_length].item()]

    def _f_min(self, input_ids: torch.LongTensor):
        """Get the previous token min."""
        return min(self.prf[input_ids[-1 - i].item()] for i in range(0, self.config.prefix_length))
    
    def get_greenlist_ids(self, input_ids: torch.LongTensor) -> list[int]:
        """Get greenlist ids for the input_ids."""
        return self.window_scheme_map[self.config.window_scheme](input_ids)
    
    def _get_greenlist_ids_left(self, input_ids: torch.LongTensor) -> list[int]:
        """Get greenlist ids for the input_ids via leftHash scheme."""
        self.rng.manual_seed((self.config.hash_key * self._f(input_ids)) % self.config.vocab_size)
        greenlist_size = int(self.config.vocab_size * self.config.gamma)
        vocab_permutation = torch.randperm(self.config.vocab_size, device=input_ids.device, generator=self.rng)
        greenlist_ids = vocab_permutation[:greenlist_size]
        return greenlist_ids
    
    def _get_greenlist_ids_self(self, input_ids: torch.LongTensor) -> list[int]:
        """Get greenlist ids for the input_ids via selfHash scheme."""
        greenlist_size = int(self.config.vocab_size * self.config.gamma)
        greenlist_ids = []
        f_x = self._f(input_ids)
        for k in range(0, self.config.vocab_size):
            h_k = f_x * int(self.prf[k])
            self.rng.manual_seed(h_k % self.config.vocab_size)
            vocab_permutation = torch.randperm(self.config.vocab_size, device=input_ids.device, generator=self.rng)
            temp_greenlist_ids = vocab_permutation[:greenlist_size]
            if k in temp_greenlist_ids:
                greenlist_ids.append(k)
        return greenlist_ids
    
    def _compute_z_score(self, observed_count: int , T: int) -> float: 
        """Compute z-score for the given observed count and total tokens."""
        expected_count = self.config.gamma
        numer = observed_count - expected_count * T 
        denom = sqrt(T * expected_count * (1 - expected_count))  
        z = numer / denom
        return z
    
    def score_sequence(self, input_ids: torch.Tensor) -> tuple[float, list[int]]:
        """Score the input_ids and return z_score and green_token_flags."""
        num_tokens_scored = len(input_ids) - self.config.prefix_length
        if num_tokens_scored < 1:
            raise ValueError(
                (
                    f"Must have at least {1} token to score after "
                    f"the first min_prefix_len={self.config.prefix_length} tokens required by the seeding scheme."
                )
            )

        green_token_count = 0
        green_token_flags = [-1 for _ in range(self.config.prefix_length)]

        for idx in range(self.config.prefix_length, len(input_ids)):
            curr_token = input_ids[idx]
            greenlist_ids = self.get_greenlist_ids(input_ids[:idx])
            if curr_token in greenlist_ids:
                green_token_count += 1
                green_token_flags.append(1)
            else:
                green_token_flags.append(0)
        
        z_score = self._compute_z_score(green_token_count, num_tokens_scored)
        return z_score, green_token_flags


class KGWLogitsProcessor(LogitsProcessor):
    """LogitsProcessor for KGW algorithm, process logits to add watermark."""

    def __init__(self, config: WatermarkConfig, utils: KGWUtils, **kwargs) -> None:
        self.config = config
        self.utils = utils

    def _calc_greenlist_mask(self, scores: torch.FloatTensor, greenlist_token_ids: torch.LongTensor) -> torch.BoolTensor:
        """Calculate greenlist mask for the given scores and greenlist token ids."""
        green_tokens_mask = torch.zeros_like(scores)
        for b_idx in range(len(greenlist_token_ids)):
            green_tokens_mask[b_idx][greenlist_token_ids[b_idx]] = 1
        final_mask = green_tokens_mask.bool()
        return final_mask

    def _bias_greenlist_logits(self, scores: torch.Tensor, greenlist_mask: torch.Tensor, greenlist_bias: float) -> torch.Tensor:
        """Bias the scores for the greenlist tokens."""
        scores[greenlist_mask] = scores[greenlist_mask] + greenlist_bias
        return scores

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        """Process logits to add watermark."""
        if input_ids.shape[-1] < self.config.prefix_length:
            return scores

        batched_greenlist_ids = [None for _ in range(input_ids.shape[0])]

        for b_idx in range(input_ids.shape[0]):
            greenlist_ids = self.utils.get_greenlist_ids(input_ids[b_idx])
            batched_greenlist_ids[b_idx] = greenlist_ids

        green_tokens_mask = self._calc_greenlist_mask(scores=scores, greenlist_token_ids=batched_greenlist_ids)

        scores = self._bias_greenlist_logits(scores=scores, greenlist_mask=green_tokens_mask, greenlist_bias=self.config.delta)
        return scores
    

class KGW(BaseWatermark):
    """Top-level class for KGW algorithm."""

    def __init__(self, algorithm_config: WatermarkConfig, **kwargs) -> None:
        self.config = algorithm_config    
        self.utils = KGWUtils(self.config)
        self.logits_processor = KGWLogitsProcessor(self.config, self.utils)
    
    def generate_with_watermark(self, input: WatermarkInput, **kwargs) -> WatermarkOutput: 
        return self.generate_with_watermark_batch([input], **kwargs)[0]

    def generate_with_watermark_batch(self, inputs: list[WatermarkInput], batch_size: int = 8, **kwargs) -> list[WatermarkOutput]:
        if not inputs:
            return []

        if self.config.tokenizer.pad_token is None and self.config.tokenizer.eos_token is not None:
            self.config.tokenizer.pad_token = self.config.tokenizer.eos_token

        prompts = [input_item.prompt for input_item in inputs]
        outputs: list[WatermarkOutput] = []
        generate_kwargs = dict(self.config.model_kwargs)
        generate_kwargs.update(kwargs)

        generate = partial(
            self.config.model.generate,
            logits_processor=LogitsProcessorList([self.logits_processor]),
            **generate_kwargs,
        )

        for start in range(0, len(prompts), batch_size):
            batch_prompts = prompts[start:start + batch_size]
            encoded_prompt = self.config.tokenizer(
                batch_prompts,
                return_tensors="pt",
                add_special_tokens=True,
                padding=True,
            ).to(self.config.device)
            encoded_watermarked_text = generate(**encoded_prompt)
            watermarked_texts = self.config.tokenizer.batch_decode(encoded_watermarked_text, skip_special_tokens=True)
            outputs.extend(WatermarkOutput(text=text) for text in watermarked_texts)

        return outputs
    
    def detect(self, text:str, **kwargs) -> DetectResult:
        return self.detect_batch([text], **kwargs)[0]

    def detect_batch(self, texts: list[str], **kwargs) -> list[DetectResult]:
        if not texts:
            return []

        if self.config.tokenizer.pad_token is None and self.config.tokenizer.eos_token is not None:
            self.config.tokenizer.pad_token = self.config.tokenizer.eos_token

        batch_size = kwargs.pop("batch_size", 32)
        results: list[DetectResult] = []

        for start in range(0, len(texts), batch_size):
            batch_texts = texts[start:start + batch_size]
            encoded_batch = self.config.tokenizer(
                batch_texts,
                return_tensors="pt",
                add_special_tokens=False,
                padding=True,
            )
            input_ids = encoded_batch["input_ids"].to(self.config.device)
            attention_mask = encoded_batch["attention_mask"]

            for idx in range(input_ids.shape[0]):
                valid_length = int(attention_mask[idx].sum().item())
                sequence = input_ids[idx][:valid_length]
                try:
                    z_score, _ = self.utils.score_sequence(sequence)
                    results.append(
                        DetectResult(
                            is_watermarked=z_score > self.config.z_threshold,
                            args={"z_score": z_score},
                        )
                    )
                except ValueError:
                    results.append(
                        DetectResult(
                            is_watermarked=False,
                            args={"z_score": None},
                        )
                    )

        return results
        