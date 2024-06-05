import streamlit as st
import subprocess
import grpc
import text_classifier_pb2
import text_classifier_pb2_grpc
import time
import psutil
import pandas as pd

# --- gRPC Client Function (with Metrics) ---

def classify_text(text):
    start_time = time.perf_counter()  # Record start time
    with grpc.insecure_channel('localhost:50051') as channel:
        stub = text_classifier_pb2_grpc.TextClassifierStub(channel)
        request = text_classifier_pb2.TextRequest(text=text)
        response = stub.ClassifyText(request)
    end_time = time.perf_counter()  # Record end time

    # Calculate Metrics
    handler_time = response.handler_time  
    response_time = end_time - start_time
    delay = response_time - handler_time 
    memory_usage = psutil.Process().memory_info().rss

    return response.category, response.confidence, handler_time, response_time, delay, memory_usage

# --- Global Variables ---
server_process = None
client_process = None
metrics_file = "model_metrics.csv"

# --- Streamlit UI ---

# Launch Server Button
if st.button("Launch Server"):
    server_process = subprocess.Popen(["python", "text_classifier_server.py"])
    st.success("Server launched!")
    
    # Automatically launch client when server starts
    client_process = subprocess.Popen(["python", "text_classifier_client.py"])  
    st.success("Client launched!")

# Stop Server Button (Disabled initially)
if st.button("Stop Server", disabled=server_process is None):
    server_process.terminate()
    server_process = None
    client_process.terminate()  # Stop the client too
    client_process = None
    st.success("Server and client stopped!")

# Text File Input
uploaded_file = st.file_uploader("Choose a text file", type=["txt"])

if uploaded_file is not None:
    # Read and display text from the uploaded file
    text = uploaded_file.read().decode('utf-8')
    st.subheader("Uploaded Text:")
    st.write(text)
    

    # Classify Text Button (Enabled when server is running and file is uploaded)
    if st.button("Classify Text", disabled=server_process is None):
        try:
            category, confidence, handler_time, response_time, delay, memory_usage = classify_text(text)
            st.write(f"Category: {category}, Confidence: {confidence:.2f}")

            # Store Metrics
            with open(metrics_file, "a") as f:  # Append to file
                f.write(f"{time.time()},{handler_time},{response_time},{delay},{memory_usage}\n")

            # Display Model Metrics from File
            st.subheader("Model Metrics:")
            try:
                df = pd.read_csv(metrics_file, names=["Timestamp", "Handler Time", "Response Time", "Delay", "Memory Usage"])
                st.dataframe(df)  # Display as a table
            except pd.errors.EmptyDataError:
                st.write("No metrics data yet.")

        except grpc.RpcError as e:
            st.error(f"Error connecting to the server: {e.details()}")

