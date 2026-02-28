from configs.base import WatermarkConfig

class KGWConfig(WatermarkConfig):
    def __init__(self, model_config, **kwargs):
        super().__init__(model_config, **kwargs)

    def load_default_config(self):
        '''
        Load default configuration values for KGW algorithm.
        '''
        self.alg_name: str = "KGW"
        self.gamma: float = 0.5
        self.delta: float = 2.0
        self.hash_key: int = 15485863
        self.prefix_length: int = 1
        self.z_threshold: float = 4.0
        self.f_scheme: str = "time"
        self.window_scheme: str = "left"
