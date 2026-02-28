'''
Data structures for watermarking inputs, outputs, and detection results.
Aiming for furthur extension to support more complex scenarios.
'''


from dataclasses import dataclass


@dataclass
class WatermarkInput:
    prompt: str

@dataclass
class WatermarkOutput:
    text: str
    args: dict | None = None

@dataclass
class DetectResult:
    is_watermarked: bool
    args: dict | None = None