# Library imports
import numpy as np
import streamlit as st
import cv2
import torch
import torchvision 
import os

# Load the Model
# Placeholder for model loading
model = torch.load('path_to_your_model')

# Class names
CLASS_NAMES = ['Corn-Common_rust', 'Potato-Early_blight', 'Tomato-Bacterial_spot']

# Set up the page configuration
st.set_page_config(page_title="Plant Disease Detection", layout="wide")

# Custom CSS
st.markdown("""
    <style>
    .big-font {
        font-size:18px !important;
    }
    .title {
        font-size:24px !important;
        font-weight: bold;
    }
    .prediction {
        color: #ff4b4b;
        font-size:20px;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

# Title and Description
st.title("Plant Disease Detection")
st.markdown("<div class='title'>Detect Diseases in Plant Leaves</div>", unsafe_allow_html=True)
st.markdown("<div class='big-font'>Upload an image of the plant leaf and get instant predictions.</div>", unsafe_allow_html=True)

# Sidebar
st.sidebar.title("About")
st.sidebar.info("This application is a deep learning tool to help identify diseases in plant leaves.")

# Upload Image
plant_image = st.file_uploader("Choose an image...", type="jpg")
submit = st.button('Predict')

# On predict button click
if submit:
    if plant_image is not None:
        # Convert the file to an opencv image
        file_bytes = np.asarray(bytearray(plant_image.read()), dtype=np.uint8)
        opencv_image = cv2.imdecode(file_bytes, 1)

        # Displaying the image
        st.image(opencv_image, channels="BGR", use_column_width=True)
        st.write(f"Image Dimensions: {opencv_image.shape[0]} x {opencv_image.shape[1]} pixels")
        
        # Resize and preprocess the image
        opencv_image = cv2.resize(opencv_image, (256, 256))
        opencv_image = np.expand_dims(opencv_image, axis=0)  # Add batch dimension

        # Make Prediction
        with st.spinner('Predicting...'):
            Y_pred = model.predict(opencv_image)
            result = CLASS_NAMES[np.argmax(Y_pred)]
            st.markdown(f"<div class='prediction'>Predicted: {result.split('-')[0]} leaf with {result.split('-')[1]}</div>", unsafe_allow_html=True)
