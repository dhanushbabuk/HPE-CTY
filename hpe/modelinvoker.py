from .torch_serve import TorchServeManager
from .vllm import VLLMManager


class ModelInvoker:
    def __init__(self, servingengine):
        self.serving_engine = servingengine
        self.manager = None

    def invoke(self, model_name, model_file):
        if self.serving_engine == 'torchserve':
            self.manager = TorchServeManager()
            self.invoke_torchserve(model_name, model_file)
        elif self.serving_engine == 'vLLM':
            self.manager = VLLMManager()
            self.invoke_vLLM(model_name, model_file)
        else:
            raise ValueError(f"Invalid tunable: {self.serving_engine}")

    def invoke_torchserve(self, model_name, model_file):
        self.manager.start_torchserve(model_name, model_file)

    def invoke_vLLM(self, model_name, model_file):
        self.manager.start_vllm(model_name, model_file)
