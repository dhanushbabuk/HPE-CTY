# modelinvoker.py

from .torch_serve import TorchServeManager
from .vllm import VLLMManager

class ModelInvoker:
    def __init__(self, serving_engine):
        self.serving_engine = serving_engine
        self.manager = self._get_manager(serving_engine)

    def _get_manager(self, serving_engine):
        if serving_engine.lower() == 'torchserve':
            return TorchServeManager()
        elif serving_engine.lower() == 'vllm':
            return VLLMManager()
        else:
            raise ValueError(f"Unsupported serving engine: {serving_engine}")

    def invoke(self, model_name, model_file):
        self.manager.start(model_name, model_file)

    def stop(self):
        self.manager.stop()

