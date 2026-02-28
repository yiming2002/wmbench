'''
Base pipeline for watermarking algorithms. 
It dynamically loads the specified watermarking algorithm and its configuration, 
and provides a unified interface for generating watermarked/unwatermarked text and detecting watermarks.
'''


import importlib
from transformers import AutoConfig

from data_structure.base import WatermarkInput, WatermarkOutput, DetectResult
from configs.base import WatermarkConfig
from utils.model_loader import ModelLoader

NAME_TO_LIBRARY = {
    'KGW': ['watermarks.kgw.KGW', 'configs.kgw.KGWConfig'],
}


def import_library_from_watermark_name(name):
    if name in NAME_TO_LIBRARY:
        return NAME_TO_LIBRARY[name][0], NAME_TO_LIBRARY[name][1]
    else:
        return NAME_TO_LIBRARY[0][0], NAME_TO_LIBRARY[0][1]

class WatermarkBasePipeline:
    def __init__(self, model_loader: ModelLoader, alg_name: str = "KGW", **kwargs):
        self.alg_name = alg_name
        
        alg_library, config_library = import_library_from_watermark_name(self.alg_name)

        config_module = importlib.import_module(config_library.rsplit('.', 1)[0])
        config_class = getattr(config_module, config_library.split('.')[-1])
        config_instance = config_class(model_loader, **kwargs)


        alg_module = importlib.import_module(alg_library.rsplit('.', 1)[0])
        alg_class = getattr(alg_module, alg_library.split('.')[-1])
        self.watermark_instance = alg_class(config_instance)
        

    def generate_with_watermark(self, input: WatermarkInput, **kwargs) -> WatermarkOutput:
        return self.watermark_instance.generate_with_watermark(input, **kwargs)

    def generate_with_watermark_batch(self, inputs: list[WatermarkInput], **kwargs) -> list[WatermarkOutput]:
        return self.watermark_instance.generate_with_watermark_batch(inputs, **kwargs)
    
    def generate_without_watermark(self, input: WatermarkInput, **kwargs) -> WatermarkOutput:
        return self.watermark_instance.generate_without_watermark(input, **kwargs)

    def generate_without_watermark_batch(self, inputs: list[WatermarkInput], **kwargs) -> list[WatermarkOutput]:
        return self.watermark_instance.generate_without_watermark_batch(inputs, **kwargs)
    
    def detect(self, text: str, **kwargs) -> DetectResult:
        return self.watermark_instance.detect(text, **kwargs)

    def detect_batch(self, texts: list[str], **kwargs) -> list[DetectResult]:
        return self.watermark_instance.detect_batch(texts, **kwargs)
        