import os
import re
import streamlit as st
from datetime import datetime
from hpe.modelinvoker import ModelInvoker
from PIL import Image
import tempfile
import requests


MODEL_FILES = {
    "resnet18": "resnet18.mar",
    "bert": "bert.mar"
}

# Function to extract metrics from two lines for resnet18
def extract_resnet_metrics(lines):
    metrics = {'HandlerTime.ms': None, 'PredictionTime.ms': None, 'Timestamp': None}
    for line in lines:
        handler_time_match = re.search(r'HandlerTime.ms:([\d.]+)', line)
        prediction_time_match = re.search(r'PredictionTime.ms:([\d.]+)', line)
        timestamp_match = re.search(r'timestamp:(\d+)', line)
        if handler_time_match:
            metrics['HandlerTime.ms'] = float(handler_time_match.group(1))
        if prediction_time_match:
            metrics['PredictionTime.ms'] = float(prediction_time_match.group(1))
        if timestamp_match:
            timestamp = int(timestamp_match.group(1))
            metrics['Timestamp'] = datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S")
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
        metrics['Timestamp'] = datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S")
    return metrics

# Retrieve and display model metrics
def display_metrics():
    log_file_path = 'logs/model_metrics.log'
    if not os.path.exists(log_file_path):
        st.error(f"Log file not found: {log_file_path}")
        return

    with open(log_file_path, 'r') as file:
        lines = file.readlines()

    resnet_metrics = []
    count_resnet = 0
    for i in range(len(lines) - 1, -1, -1):
        if "resnet18" in lines[i]:
            metrics = extract_resnet_metrics(lines[i-1:i+1])
            resnet_metrics.append(metrics)
            count_resnet += 1
            if count_resnet == 5:
                break

    bert_metrics = []
    count_bert = 0
    for i in range(len(lines) - 1, -1, -1):
        if "bert" in lines[i]:
            metrics = extract_bert_metrics(lines[i])
            bert_metrics.append(metrics)
            count_bert += 1
            if count_bert == 5:
                break

    st.write("Metrics for resnet18:")
    for metrics in reversed(resnet_metrics):
        handler_time = metrics.get('HandlerTime.ms', 'N/A')
        prediction_time = metrics.get('PredictionTime.ms', 'N/A')
        timestamp = metrics.get('Timestamp', 'N/A')
        st.write(f"Timestamp: {timestamp} | Handler Time: {handler_time} ms | Prediction Time: {prediction_time} ms")

    st.write("Prediction Times for bert:")
    for metrics in reversed(bert_metrics):
        prediction_time = metrics.get('PredictionTime.ms', 'N/A')
        timestamp = metrics.get('Timestamp', 'N/A')
        if prediction_time != 'N/A':
            st.write(f"Timestamp: {timestamp} | Prediction Time: {prediction_time} ms")

# Main function
def main():
    st.title("ML Serving Framework")

    # Model selection dropdown in the sidebar
    serving_engine_choice = st.sidebar.selectbox("Select ML Serving Engine", ["None", "TorchServe", "VLLM"])

    # Model selection dropdown in the sidebar
    model_choice = st.sidebar.selectbox("Select Model", ["None", "ResNet18", "BERT"])

    # Initialize session state for model choices
    if "last_model_choice" not in st.session_state:
        st.session_state["last_model_choice"] = None
    if "previous_model_choice" not in st.session_state:
        st.session_state["previous_model_choice"] = None

    if st.session_state["last_model_choice"] != model_choice:
        st.session_state["previous_model_choice"] = st.session_state["last_model_choice"]
        st.session_state["last_model_choice"] = model_choice

    if serving_engine_choice != "None" and model_choice != "None":
        model_invoker = ModelInvoker(serving_engine_choice)

        if st.session_state["previous_model_choice"] != model_choice:
            model_invoker.stop()
            model_invoker.invoke(model_choice, MODEL_FILES[model_choice.lower()])

        if model_choice == "ResNet18":
            st.sidebar.header("ResNet Model")
            uploaded_images = st.sidebar.file_uploader("Upload images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
            if uploaded_images:
                st.header("ResNet Model")
                access_images(uploaded_images)
                invoke_resnet(uploaded_images)

        elif model_choice == "BERT":
            st.sidebar.header("BERT Model")
            uploaded_file = st.sidebar.file_uploader("Upload a .txt file", type=["txt"])
            if uploaded_file:
                text = uploaded_file.read().decode("utf-8")
                st.header("BERT Model")
                st.write(f"Text in the uploaded file: {text}")
                invoke_bert(text)

    if st.sidebar.button("Show Model Metrics"):
        display_metrics()

def access_images(images):
    st.write("Accessing uploaded images:")
    for image in images:
        img = Image.open(image)
        st.image(img, caption=image.name)
    return images

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
                st.write(f"Failed to get prediction for {image.name}: {response.status_code}, {response.text}")
        except requests.exceptions.RequestException as e:
            st.error(f"Failed to connect to TorchServe: {e}")

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
                    st.write(f"Failed to get prediction for input: {response.status_code}, {response.text}")
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to connect to TorchServe: {e}")
    finally:
        os.remove(temp_file_path)

if __name__ == "__main__":
    main()
