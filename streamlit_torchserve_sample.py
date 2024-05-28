#reference open source snippets
# Sample code for streamlit and hhp / REST API

import streamlit as st
import requests  # REST API
import time
import os
import glob
from PIL import Image
import subprocess

#Check if torchserve is running
def is_torchserve_running():
    try:
        ts_addr = "http://localhost:8080/ping"  #change port number if needed
        response = requests.get(ts_addr) # uses default port number
        if response.status_code == 200:  #200 is the HTTP status code for "OK", a success indicating server is running
            return True
        else:
            return False
    except requeste.exceptions.ConnectionError:
        print ("Connection Error")
        return False


#launch torchserve
def launch_torchserve():
    running = False
    #fetch_params() - implement params to be fetched. currently, it is hard coded in os.system call
    #"Start torchserve. Check if already running"
    if (is_torchserve_running()):
        running = True
        st.success("Torch serve already running")
        return running
    try:  #Note for now, the params are hardcoded. need to pich them from fetch_params. All params can be a simple yaml file
        os.system ("torchserve --start --ncs --model-store model-store --models resnet18=resnet-18.mar") # e.g. model mar, model paths etc. are params that can be obtained from user or from a configuration file such as a yaml file
        st.success("torchserve started...")
    except Exception as e:
        st.error(f"Failed to start torchserve {e}")



# Function to launch TorchServe

# Access and display .mar models

def display_models(mar_path):
    st.write("Displaying .mar models from:", mar_path)
    models = glob.glob(f"{mar_path}/*.mar")  # This gets all .mar files in the directory
    for model in models:
        st.write(model)

# Access images from a folder path
def access_images(image_folder_path):
    st.write("Accessing images from:", image_folder_path)
    images = glob.glob(f"{image_folder_path}/*.jpg")  # This gets all .jpg files in the directory
    for image in images:
        st.image(image)  # This displays the image
    return images

# Initiate TorchServe to serve, and display predicted images
def invoke_torchserve(images):
    st.write("Invoking TorchServe and displaying predicted images...")

     # Prediction: Hardcoded to kitten.jpg as an example in shell command below. Change this to API 
      #Use images in a folder path and enhance the code as in the Placeholder section below      
    shell_cmd="curl -X POST http://localhost:8080/predictions/resnet18 -T ./images/kitten.jpg"
    result = subprocess.check_output(shell_cmd, shell=True)
    ## Fix this - this is an API call
    #response = requests.post('http://localhost:8080/predictions/resnet18 -T kitten.jpg') 
    st.write(f"Prediction for kitten: {result}")  # This displays the prediction
    #st.write(f"Prediction for {image}: {response.json()}")  # This displays the prediction
    
    #Place holder: Update this code. this will loop across images in a folder and fire predictions
    
    #for image in images:
        #response = requests.post('http://localhost:8080/predictions/resnet18 -T', files={"data": open(image, 'rb')})
        #Hardcoded to kitten.jpg as an example. Use the image in the for loop above as a parameter to the call below. 
        #st.write(f"Prediction for {image}: {response.json()}")  # This displays the prediction

    #Here, we will add in metrics. The python code you are writing can be invoked from here
    def obtain_metrics():
        return 0  # Need to add the metrics collection portiion here.

# Main function
def main():
    # Access serving frameworks list - Torchserve, vLLM,. User will select TorchServe. 
    # Select framework
    # Start serving choosing the selected framework
    st.title("TorchServe ML Serving framework")

    # Button to launch TorchServe
    if st.button("Launch TorchServe"):
        launch_torchserve()

    # Input field for .mar model path
    mar_path = st.text_input("Enter .mar model path:")
    if mar_path:
        display_models(mar_path)

    # Input field for image folder path
    image_folder_path = st.text_input("Enter image folder path:")
    if image_folder_path:
        images = access_images(image_folder_path)
        if images:
            invoke_torchserve(images)
    #obtain metrics

# Run the main function
if __name__ == "__main__":
    main()
