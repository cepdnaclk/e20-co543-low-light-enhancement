import torch
import streamlit as st
import numpy as np
import cv2
import os
from PIL import Image
from utils.image_processing import adjust_brightness, gamma_transform, apply_noise_reduction, sharpen_image, log_transform, apply_median_blur
from models.model_loader import load_model, postprocess, preprocess


print("is Cuda available : ", torch.cuda.is_available())
# Load the Zero-DCE model
DCE_net = load_model()


# Set Streamlit page config
st.set_page_config(page_title="Low Light Image Enhancement", layout="wide")

# Ensure output directory exists
os.makedirs("outputs", exist_ok=True)

# Sidebar - File Upload
st.sidebar.header("Upload Image")
uploaded_file = st.sidebar.file_uploader("Choose an image...", type=["jpg", "png", "jpeg", "ARW"])

# Mode Selection
view_mode = st.sidebar.radio("View Mode", ["Output Mode", "Compare Mode"], index=0)

if uploaded_file:
    image = Image.open(uploaded_file)
    image = np.array(image)

    # Reset settings when a new image is uploaded
    if "last_uploaded_file" not in st.session_state or uploaded_file.name != st.session_state["last_uploaded_file"]:
      st.session_state["brightness"] = 0
      st.session_state["apply_gamma"] = False
      st.session_state["gamma_value"] = 1.0
      st.session_state["noise_reduction"] = 0
      st.session_state["sharpening"] = 0
      st.session_state["apply_log"] = False
      st.session_state["auto_adjust"] = False
      st.session_state["median_blur_strength"] = 0
      st.session_state["last_uploaded_file"] = uploaded_file.name  # Update last uploaded file

    # Sidebar - Image Adjustments
    st.sidebar.header("Adjustments")
    brightness = st.sidebar.slider("Brightness", -100, 100, 0)
    apply_gamma = st.sidebar.checkbox("Apply Gamma Correction")
    gamma_value = st.sidebar.number_input("Gamma Value", min_value=0.1, value=1.0, step=0.1, format="%.1f")
    noise_reduction = st.sidebar.slider("Noise Reduction(Bilateral)", 0, 5, 0)
    median_blur_strength = st.sidebar.slider("Noise Reduction(Median)", 0, 5, 0)
    sharpening = st.sidebar.slider("Sharpening", 0, 10, 0)
    apply_log = st.sidebar.checkbox("Log Transformation")
    auto_adjust = st.sidebar.checkbox("Auto Adjust (AI Model)")

    # TODO: reset values when uploading a new file
    
    # brightness = st.sidebar.slider("Brightness", -100, 100, st.session_state.get("brightness", 0), key="brightness")
    # apply_gamma = st.sidebar.checkbox("Apply Gamma Correction", st.session_state.get("apply_gamma", False), key="apply_gamma")
    # gamma_value = st.sidebar.number_input("Gamma Value", min_value=0.1, value=st.session_state.get("gamma_value", 1.0), step=0.1, format="%.1f", key="gamma_value")
    # noise_reduction = st.sidebar.slider("Noise Reduction(Bilateral)", 0, 10, st.session_state.get("noise_reduction", 0), key="noise_reduction")
    # median_blur_strength = st.sidebar.slider("Noise Reduction(Median)", 0, 5, 0, st.session_state.get("noise_reduction", 0), key="median_blur_strength")
    # sharpening = st.sidebar.slider("Sharpening", 0, 10, st.session_state.get("sharpening", 0), key="sharpening")
    # apply_log = st.sidebar.checkbox("Log Transformation", st.session_state.get("apply_log", False), key="apply_log")
    # auto_adjust = st.sidebar.checkbox("Auto Adjust (AI Model)", st.session_state.get("auto_adjust", False), key="auto_adjust")
    # median_blur_strength = st.sidebar.slider("Median Blur (Noise Reduction)", 0, 5, st.session_state.get("median_blur_strength", 0), key="median_blur_strength")


    # Process Image
    processed_image = image.copy()
    processed_image = adjust_brightness(processed_image, brightness)
    if apply_gamma:
        processed_image = gamma_transform(processed_image, gamma_value)
    if apply_log:
        processed_image = log_transform(processed_image)
    if noise_reduction > 0:
        processed_image = apply_noise_reduction(processed_image, noise_reduction)
    if sharpening > 0:
        processed_image = sharpen_image(processed_image, sharpening)
    # if auto_adjust:
    #     processed_image = preprocess(processed_image)
    #     _,processed_image,_ = DCE_net(processed_image)
    #     processed_image = postprocess(processed_image)
    if auto_adjust:
        # Set the device and move the model to GPU if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        DCE_net.to(device)

        # Preprocess the image and move the tensor to the same device
        input_tensor = preprocess(processed_image).to(device)

        # Run the model inference
        _, processed_tensor, _ = DCE_net(input_tensor)

        # Postprocess the output back to a displayable image
        processed_image = postprocess(processed_tensor)
    if median_blur_strength > 0:
      processed_image = apply_median_blur(processed_image, median_blur_strength)


    # Display Images
    col1, col2 = st.columns(2)
    if view_mode == "Compare Mode":
        col1.image(image, caption="Original Image", use_container_width=True)
        col2.image(processed_image, caption="Enhanced Image", use_container_width=True)
    else:
        st.image(processed_image, caption="Enhanced Image", use_container_width=True)

    # Save & Download Button
    save_path = os.path.join("outputs", "enhanced_image.png")
    Image.fromarray(processed_image).save(save_path)
    with open(save_path, "rb") as file:
        st.download_button("Download Enhanced Image", file, file_name="enhanced_image.png", mime="image/png")
