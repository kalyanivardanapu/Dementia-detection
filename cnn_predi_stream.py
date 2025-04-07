import streamlit as st
import numpy as np
import cv2
import base64
from tensorflow.keras.models import load_model

# Load the trained model
model_path = 'cnn_model/cnn_model_finetuned.h5'  # Use forward slashes for compatibility
model = load_model(model_path)

# Define class labels
CLASS_LABELS = {
    0: "Mild Demented",
    1: "Moderate Demented",
    2: "Non-Demented",
    3: "Very Mild Demented"
}

# Function to set background from local image
def set_bg_from_local(image_path):
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpg;base64,{encoded_string}");
            background-size: cover;
            background-attachment: fixed;
            background-position: center;
        }}
        .blue-text {{
            color: #003366;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Set background image
set_bg_from_local("background.jpeg")  # <-- Change this if your image name is different

# Preprocessing function
def preprocess_image(image):
    image = cv2.resize(image, (50, 50))  # Resize to match model input
    image = image / 255.0               # Normalize the image
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# UI Title and Instructions
st.markdown('<h1 class="blue-text">🧠 Dementia Detection with CNN</h1>', unsafe_allow_html=True)
st.markdown('<p class="blue-text">Upload an MRI image to predict the dementia type.</p>', unsafe_allow_html=True)
st.markdown("""
<div class='blue-text'>

</div>
""", unsafe_allow_html=True)

# Upload section
uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    image_resized = cv2.resize(image, (300, 300))
    st.image(image_resized, channels="BGR", caption="Uploaded Image", use_container_width=False)

    processed_image = preprocess_image(image)

    prediction = model.predict(processed_image)
    predicted_class = np.argmax(prediction, axis=1)[0]
    confidence = prediction[0][predicted_class] * 100

    st.markdown(f'<h3 class="blue-text">🧪 Prediction: {CLASS_LABELS[predicted_class]}</h3>', unsafe_allow_html=True)
    st.markdown(f'<h3 class="blue-text">✅ Confidence: {confidence:.2f}%</h3>', unsafe_allow_html=True)

    if predicted_class == 0:
        st.markdown("""
        <div class='blue-text'>
        <h4>Recommendations for Mild Demented:</h4>
        <ul>
            <li>Engage in brain-stimulating activities like puzzles and memory games.</li>
            <li>Maintain a consistent daily routine to reduce confusion.</li>
            <li>Prioritize regular physical exercise and a balanced diet.</li>
            <li>Schedule regular follow-ups with a healthcare professional.</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    elif predicted_class == 1:
        st.markdown("""
        <div class='blue-text'>
        <h4>Recommendations for Moderate Demented:</h4>
        <ul>
            <li>Seek immediate medical consultation for advanced care options.</li>
            <li>Ensure a safe living environment with minimal hazards.</li>
            <li>Provide emotional and physical support to reduce agitation.</li>
            <li>Maintain a structured routine and involve caregivers actively.</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    elif predicted_class == 2:
        st.markdown("""
        <div class='blue-text'>
        <h4>Recommendations for Non-Demented:</h4>
        <ul>
            <li>Continue a healthy lifestyle to maintain brain health.</li>
            <li>Engage in physical activities like walking or yoga regularly.</li>
            <li>Follow a diet rich in antioxidants and omega-3 fatty acids.</li>
            <li>Manage stress levels through relaxation techniques.</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    elif predicted_class == 3:
        st.markdown("""
        <div class='blue-text'>
        <h4>Recommendations for Very Mild Demented:</h4>
        <ul>
            <li>Monitor symptoms and seek early interventions to prevent progression.</li>
            <li>Practice mental exercises to boost memory and cognitive skills.</li>
            <li>Maintain social interactions and engage in group activities.</li>
            <li>Consult a doctor for preventive strategies and symptom tracking.</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<hr>", unsafe_allow_html=True)
st.markdown('<p class="blue-text">Developed for early detection of dementia using deep learning techniques.</p>', unsafe_allow_html=True)