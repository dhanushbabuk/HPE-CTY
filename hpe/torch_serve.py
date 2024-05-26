import os
import subprocess
import time

import requests

MODEL_STORE = "model_store"
MODEL_FILES = {
    "resnet18": "resnet-18.mar",
    "bert": "bert.mar"
}


class TorchServeManager:
    def __init__(self):
        self.current_model = None

    def is_torchserve_running(self):
        try:
            ts_addr = "http://localhost:8080/ping"
            response = requests.get(ts_addr)
            if response.status_code == 200:
                return True
            else:
                return False
        except requests.exceptions.ConnectionError:
            return False

    def start_torchserve(self, model_name, model_file):
        if self.is_torchserve_running():
            print("TorchServe is already running")
            return

        try:
            print("Starting TorchServe...")
            os.system(
                f"torchserve --start --ncs --model-store {MODEL_STORE} --models {model_name.lower()}={model_file}")
            # subprocess.run(["torchserve", "--start", "--ncs",
            #                "--model-store", model_path])
            self.current_model = model_name
            time.sleep(5)
        except Exception as e:
            print(f"Failed to start TorchServe: {e}")

        if self.is_torchserve_running():
            print("TorchServe started successfully")
        else:
            print("TorchServe failed to start")

    def stop_torchserve(self):
        if not self.is_torchserve_running():
            print("TorchServe is not running")
            return

        try:
            print("Stopping TorchServe...")
            subprocess.run(["torchserve", "--stop"])
            self.current_model = None
            time.sleep(5)
        except Exception as e:
            print(f"Failed to stop TorchServe: {e}")

        if not self.is_torchserve_running():
            print("TorchServe stopped successfully")
        else:
            print("TorchServe failed to stop")
