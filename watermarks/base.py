'''
Base class for watermarking algorithms.
It defines the common interface for generating watermarked/unwatermarked text and detecting watermarks.'''


from typing import Union
from configs.base import WatermarkConfig
from data_structure.base import WatermarkInput, WatermarkOutput, DetectResult 
from utils.model_loader import ModelLoader


class BaseWatermark:
    def __init__(self, config: WatermarkConfig, **kwargs) -> None:
        self.config = config

    def generate_with_watermark(self, input: WatermarkInput, **kwargs) -> WatermarkOutput: 
        pass

    def generate_with_watermark_batch(self, inputs: list[WatermarkInput], **kwargs) -> list[WatermarkOutput]:
        return [self.generate_with_watermark(input_item, **kwargs) for input_item in inputs]

    def generate_without_watermark(self, input: WatermarkInput, **kwargs) -> WatermarkOutput:
        return self.generate_without_watermark_batch([input], **kwargs)[0]

    def generate_without_watermark_batch(self, inputs: list[WatermarkInput], batch_size: int = 8, **kwargs) -> list[WatermarkOutput]:
        if not inputs:
            return []

        if self.config.tokenizer.pad_token is None and self.config.tokenizer.eos_token is not None:
            self.config.tokenizer.pad_token = self.config.tokenizer.eos_token

        prompts = [input_item.prompt for input_item in inputs]
        outputs: list[WatermarkOutput] = []
        generate_kwargs = dict(self.config.model_kwargs)
        generate_kwargs.update(kwargs)

        for start in range(0, len(prompts), batch_size):
            batch_prompts = prompts[start:start + batch_size]
            encoded_prompt = self.config.tokenizer(
                batch_prompts,
                return_tensors="pt",
                add_special_tokens=True,
                padding=True,
            ).to(self.config.device)
            encoded_unwatermarked_text = self.config.model.generate(**encoded_prompt, **generate_kwargs)
            unwatermarked_texts = self.config.tokenizer.batch_decode(encoded_unwatermarked_text, skip_special_tokens=True)
            outputs.extend(WatermarkOutput(text=text) for text in unwatermarked_texts)

        return outputs

    def detect(self, text:str, **kwargs) -> DetectResult:
        pass

    def detect_batch(self, texts: list[str], **kwargs) -> list[DetectResult]:
        return [self.detect(text, **kwargs) for text in texts]




