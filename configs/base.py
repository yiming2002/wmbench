from utils.model_loader import ModelLoader


class WatermarkConfig:
    '''
    Base configuration class for watermarking algorithms.
    '''
    def __init__(self, model_config: ModelLoader, **kwargs):
        '''
        Initialize the configuration with model-related parameters.
        '''
        self.model = model_config.model
        self.tokenizer = model_config.tokenizer
        self.vocab_size = model_config.vocab_size
        self.device = model_config.device
        self.model_kwargs = model_config.model_kwargs
        
        '''
        Initialize the configuration with algorithm-related parameters.
        '''
        self.load_default_config()
        if kwargs:
            self.load_customize_config(kwargs)

    def load_default_config(self):
        pass
    
    def load_customize_config(self, kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
