# modelinvoker.py

from .torch_serve import TorchServeManager
from .vllm import VLLMManager
import streamlit as st
from PIL import Image
import requests
import tempfile
import os

class ModelInvoker:
    def __init__(self, serving_engine):
        self.serving_engine = serving_engine
        self.manager = TorchServeManager()

    def handle_ui_and_invoke(self, model_choice):
        model_name = model_choice.lower()
        model_file = f"hpe/model_store/{model_name}.mar"

        if self.manager.is_torchserve_running() and self.manager.current_model != model_name:
            self.manager.stop()
        self.manager.start(model_name, model_file)

        if model_choice == "ResNet18":
            st.sidebar.header("ResNet Model")
            uploaded_images = st.sidebar.file_uploader("Upload images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
            if uploaded_images:
                st.header("ResNet Model")
                images = self._access_images(uploaded_images)
                self._invoke_resnet(images)

        elif model_choice == "BERT":
            st.sidebar.header("BERT Model")
            uploaded_file = st.sidebar.file_uploader("Upload a .txt file", type=["txt"])
            if uploaded_file:
                text = uploaded_file.read().decode("utf-8")
                st.header("BERT Model")
                st.write(f"Text in the uploaded file: {text}")
                self._invoke_bert(text)

    def _access_images(self, images):
        st.write("Accessing uploaded images:")
        for image in images:
            img = Image.open(image)
            st.image(img, caption=image.name)
        return images
        

    def _invoke_resnet(self, images):
        st.write("Invoking ResNet model and displaying predicted images...")
        url = "http://localhost:8080/predictions/resnet18"
        for image in images:
            image.seek(0)
            files = {"data": image}
            try:
                response = requests.post(url, files=files)
                if response.status_code == 200:
                    st.write(f"Prediction for {image.name}: {response.json()}")
                else:
                    st.write(f"Failed to get prediction for {image.name}: {response.status_code}, {response.text}")
            except requests.exceptions.RequestException as e:
                st.error(f"Failed to connect to TorchServe: {e}")

    def _invoke_bert(self, text):
        st.write("Invoking BERT model for text prediction...")
        url = "http://localhost:8080/predictions/bert"  # Assuming TorchServe is running on port 8080
        
        # Prepare data for BERT: Check if BERT expects a list of strings
        data = {"data": [text]}  

        try:
            response = requests.post(url, json=data) 
            if response.status_code == 200:
                st.write(f"Prediction for input: {response.json()}")
            else:
                st.write(f"Failed to get prediction for input: {response.status_code}, {response.text}")
        except requests.exceptions.RequestException as e:
            st.error(f"Failed to connect to TorchServe: {e}")
