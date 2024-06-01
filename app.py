import streamlit as st
from hpe.modelinvoker import ModelInvoker
from hpe.display_metrics import displayMetrics

st.title("ML Serving Framework")
serving_engine_choice = st.sidebar.selectbox("Select ML Serving Engine", ["None", "TorchServe", "vLLM"])
model_choice = st.sidebar.selectbox("Select Model", ["None", "ResNet18", "BERT"])

if serving_engine_choice != "None" and model_choice != "None":
    model_invoker = ModelInvoker(serving_engine_choice)
    model_invoker.handle_ui_and_invoke(model_choice)

if st.sidebar.button("Show Model Metrics"):
    displayMetrics.display_metrics(st)