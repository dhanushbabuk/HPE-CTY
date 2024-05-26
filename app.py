import glob
import os
import re
import subprocess
import tempfile
import time
from datetime import datetime

import requests
import streamlit as st
from PIL import Image

from .modelinvoker import ModelInvoker

# Define hardcoded parameters
MODEL_STORE = "model_store"
MODEL_FILES = {
    "resnet18": "resnet-18.mar",
    "bert": "bert.mar"
}


# Track the currently running model
CURRENT_MODEL = None

# Check if torchserve is running


def is_torchserve_running():
    try:
        ts_addr = "http://localhost:8080/ping"
        response = requests.get(ts_addr)
        if response.status_code == 200:
            return True
        else:
            return False
    except requests.exceptions.ConnectionError:
        return False


# Launch torchserve with the selected model
def launch_torchserve(model_name):
    global CURRENT_MODEL
    if is_torchserve_running():
        if CURRENT_MODEL == model_name:
            st.success("TorchServe already running with the selected model")
            return
        else:
            stop_torchserve()

    model_file = MODEL_FILES.get(model_name.lower())
    if not model_file:
        st.error(f"Model file for {model_name} not found")
        return

    try:
        with st.spinner(f"Starting TorchServe with {model_name}..."):
            os.system(
                f"torchserve --start --ncs --model-store {MODEL_STORE} --models {model_name.lower()}={model_file}")
            CURRENT_MODEL = model_name
            time.sleep(5)
    except Exception as e:
        st.error(f"Failed to start TorchServe: {e}")

    if is_torchserve_running():
        st.success(f"TorchServe started successfully with {model_name}")
    else:
        st.error(f"TorchServe failed to start with {model_name}")

# Stop torchserve


def stop_torchserve():
    global CURRENT_MODEL
    if not is_torchserve_running():
        st.success("TorchServe is not running")
        return

    try:
        with st.spinner("Stopping TorchServe..."):
            os.system("torchserve --stop")
            CURRENT_MODEL = None
            time.sleep(5)
    except Exception as e:
        st.error(f"Failed to stop TorchServe: {e}")

    if not is_torchserve_running():
        st.success("TorchServe stopped successfully")
    else:
        st.error("TorchServe failed to stop")

# Display .mar models


def display_models(mar_path):
    st.write("Displaying .mar models from:", mar_path)
    models = glob.glob(f"{mar_path}/*.mar")
    for model in models:
        st.write(model)

# Access images


def access_images(images):
    st.write("Accessing uploaded images:")
    for image in images:
        img = Image.open(image)
        st.image(img, caption=image.name)
    return images

# Invoke ResNet


def invoke_resnet(images):
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
                st.write(
                    f"Failed to get prediction for {image.name}: {response.status_code}, {response.text}")
        except requests.exceptions.RequestException as e:
            st.error(f"Failed to connect to TorchServe: {e}")

# Invoke BERT


def invoke_bert(text):
    st.write("Invoking BERT model for text prediction...")
    url = "http://localhost:8080/predictions/bert"

    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(text.encode("utf-8"))
        temp_file_path = temp_file.name

    try:
        with st.spinner("Processing BERT prediction..."):
            with open(temp_file_path, 'rb') as f:
                files = {'data': f}
                response = requests.post(url, files=files)
                st.write(f"Given text is: {text}")
                if response.status_code == 200:
                    st.write(f"Prediction for input: {response.json()}")
                else:
                    st.write(
                        f"Failed to get prediction for input: {response.status_code}, {response.text}")
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to connect to TorchServe: {e}")
    finally:
        os.remove(temp_file_path)


# Function to extract metrics from two lines for resnet18
def extract_resnet_metrics(lines):
    metrics = {'HandlerTime.ms': None,
               'PredictionTime.ms': None, 'Timestamp': None}
    for line in lines:
        handler_time_match = re.search(r'HandlerTime.ms:([\d.]+)', line)
        prediction_time_match = re.search(r'PredictionTime.ms:([\d.]+)', line)
        timestamp_match = re.search(r'timestamp:(\d+)', line)
        if handler_time_match:
            metrics['HandlerTime.ms'] = float(handler_time_match.group(1))
        if prediction_time_match:
            metrics['PredictionTime.ms'] = float(
                prediction_time_match.group(1))
        if timestamp_match:
            timestamp = int(timestamp_match.group(1))
            metrics['Timestamp'] = datetime.fromtimestamp(
                timestamp).strftime("%Y-%m-%d %H:%M:%S")
    return metrics

# Function to extract prediction time from one line for bert


def extract_bert_metrics(line):
    metrics = {'PredictionTime.ms': None, 'Timestamp': None}
    prediction_time_match = re.search(r'PredictionTime.ms:([\d.]+)', line)
    timestamp_match = re.search(r'timestamp:(\d+)', line)
    if prediction_time_match:
        metrics['PredictionTime.ms'] = float(prediction_time_match.group(1))
    if timestamp_match:
        timestamp = int(timestamp_match.group(1))
        metrics['Timestamp'] = datetime.fromtimestamp(
            timestamp).strftime("%Y-%m-%d %H:%M:%S")
    return metrics

# Retrieve and display model metrics


def display_metrics():
    # Update this with the path to your log file
    log_file_path = 'logs/model_metrics.log'
    if not os.path.exists(log_file_path):
        st.error(f"Log file not found: {log_file_path}")
        return

    with open(log_file_path, 'r') as file:
        lines = file.readlines()

    # Extract metrics for resnet18
    resnet_metrics = []
    count_resnet = 0
    for i in range(len(lines) - 1, -1, -1):
        if "resnet18" in lines[i]:
            metrics = extract_resnet_metrics(lines[i-1:i+1])
            resnet_metrics.append(metrics)
            count_resnet += 1
            if count_resnet == 5:
                break

    # Extract metrics for bert
    bert_metrics = []
    count_bert = 0
    for i in range(len(lines) - 1, -1, -1):
        if "bert" in lines[i]:
            metrics = extract_bert_metrics(lines[i])
            bert_metrics.append(metrics)
            count_bert += 1
            if count_bert == 5:
                break

    # Display the extracted model metrics
    st.write("Metrics for resnet18:")
    for metrics in reversed(resnet_metrics):
        handler_time = metrics.get('HandlerTime.ms', 'N/A')
        prediction_time = metrics.get('PredictionTime.ms', 'N/A')
        timestamp = metrics.get('Timestamp', 'N/A')
        st.write(
            f"Timestamp: {timestamp} | Handler Time: {handler_time} ms | Prediction Time: {prediction_time} ms")

    st.write("Prediction Times for bert:")
    for metrics in reversed(bert_metrics):
        prediction_time = metrics.get('PredictionTime.ms', 'N/A')
        timestamp = metrics.get('Timestamp', 'N/A')
        if prediction_time != 'N/A':
            st.write(
                f"Timestamp: {timestamp} | Prediction Time: {prediction_time} ms")


# Main function
def main():
    st.title("ML Serving Framework")

    # Model selection dropdown in the sidebar
    serving_engine_choice = st.sidebar.selectbox(
        "Select ML Serving Engine", ["None", "TorchServe", "VLLM"])

    # Model selection dropdown in the sidebar
    model_choice = st.sidebar.selectbox(
        "Select Model", ["None", "ResNet18", "BERT"])

    model_invoker = None
    if serving_engine_choice != "None":
        model_invoker = ModelInvoker(serving_engine_choice)

    # Track if the model has changed
    if st.session_state.get("last_model_choice") != model_choice:
        st.session_state["last_model_choice"] = model_choice
        if model_choice != "None":
            stop_torchserve()
            launch_torchserve(model_choice)

    # Display input fields based on selected model
    if model_choice == "ResNet18":
        st.sidebar.header("ResNet Model")
        uploaded_images = st.sidebar.file_uploader(
            "Upload images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
        if st.sidebar.button("Launch TorchServe"):
            if model_choice != "None":
                launch_torchserve(model_choice)
        if st.sidebar.button("Stop TorchServe"):
            stop_torchserve()

        if uploaded_images:
            st.header("ResNet Model")
            access_images(uploaded_images)
            invoke_resnet(uploaded_images)

    elif model_choice == "BERT":
        st.sidebar.header("BERT Model")
        uploaded_file = st.sidebar.file_uploader(
            "Upload a .txt file", type=["txt"])
        if st.sidebar.button("Launch TorchServe"):
            if model_choice != "None":
                launch_torchserve(model_choice)
        if st.sidebar.button("Stop TorchServe"):
            stop_torchserve()

        if uploaded_file:
            text = uploaded_file.read().decode("utf-8")
            st.header("BERT Model")
            # Display the text from the uploaded file
            st.write(f"Text in the uploaded file: {text}")
            invoke_bert(text)

    if st.sidebar.button("Show Model Metrics"):
        display_metrics()


if __name__ == "__main__":
    main()
