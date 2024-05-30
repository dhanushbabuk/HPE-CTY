# torch_serve.py

import os
import subprocess
import time
import requests

MODEL_STORE = "model_store"

class TorchServeManager:
    def __init__(self):
        self.current_model = None

    def is_torchserve_running(self):
        try:
            response = requests.get("http://localhost:8080/ping")
            return response.status_code == 200
        except requests.exceptions.ConnectionError:
            return False

    def start(self, model_name, model_file):
        if self.is_torchserve_running():
            if self.current_model == model_name:
                print("TorchServe already running with the selected model")
                return
            else:
                self.stop()

        try:
            print(f"Starting TorchServe with {model_name}...")
            os.system(f"torchserve --start --ncs --model-store {MODEL_STORE} --models {model_name.lower()}={model_file}")
            self.current_model = model_name
            time.sleep(5)
        except Exception as e:
            print(f"Failed to start TorchServe: {e}")

        if self.is_torchserve_running():
            print(f"TorchServe started successfully with {model_name}")
        else:
            print(f"TorchServe failed to start with {model_name}")

    def stop(self):
        if not self.is_torchserve_running():
            print("TorchServe is not running")
            return

        try:
            print("Stopping TorchServe...")
            os.system("torchserve --stop")
            self.current_model = None
            time.sleep(5)
        except Exception as e:
            print(f"Failed to stop TorchServe: {e}")

        if not self.is_torchserve_running():
            print("TorchServe stopped successfully")
        else:
            print("TorchServe failed to stop")
