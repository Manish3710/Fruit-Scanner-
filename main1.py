import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
from PIL import Image  # Replace cv2 with Pillow
import io

# Load nutrition data
nutrition_data = pd.read_csv("nutrition_info.csv")  # Ensure this CSV contains food items and their nutrition info

# Tensorflow Model Prediction
def model_prediction(test_image, confidence_threshold=90):
    model = tf.keras.models.load_model("trained_model.h5")
    #model = tf.keras.models.load_model("trained_model.keras")
    # Load and preprocess the image
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(64, 64))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])  # Convert single image to batch
    
    # Get predictions
    predictions = model.predict(input_arr)
    confidence_scores = predictions[0]  # Assuming model.predict gives confidence scores per class
    
    # Get the class with the highest confidence
    max_confidence = np.max(confidence_scores)
    predicted_class_index = np.argmax(confidence_scores)
    
    # Convert confidence threshold to decimal for comparison
    confidence_threshold_decimal = confidence_threshold / 100.0
    
    # Check if the confidence exceeds the threshold
    if max_confidence >= confidence_threshold_decimal:
        return predicted_class_index, max_confidence
    else:
        return None, max_confidence

st.set_page_config(layout="centered")

# Custom CSS for styling
st.markdown(
    """
    <style>
    body {
        background-image: url('https://example.com/your-background-image.jpg'); /* Replace with your background image URL */
        background-size: cover;
        color: black;
    }

    .full-width-img img {
        display: block;
        margin-left: auto;
        margin-right: auto;
        width: 100%;
        height: auto;
    }

    .stHeader > header {
        background-color:blue;
    }
    
    .sidebar .sidebar-content {
        background-color: rgba(0, 0, 0, 0.7);
        border-radius: 10px;
        padding: 20px;

    }
    h1, h2, h3 {
        <!--text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.7);-->
        color:black;
        <!--font-family:serif;-->
    }
    .stButton > button {
        background-color: seagreen;
        color: white;
        border-radius: 5px;
        padding: 10px 20px;
        transition: background-color 0.3s ease;
    }
    .stButton > button:hover {
        background-color: white;
        color: black;
        border:1px solid black;
    }
    .stButton > button:focus {
        background-color: white;
        color: black;
        border:1px solid black;
    }
    th {
      background-color:skyblue;
      color:blue;
    }
    td {
      background-color:#fff;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Sidebar
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page", ["Home", "Prediction"])

# Main Page
if app_mode == "Home":
    st.header("FOOD & FRUITS RECOGNITION SYSTEM")
    image_path = "food_scanner_background3.jpg"
    st.image(image_path, use_container_width=True)

# Prediction Page
elif app_mode == "Prediction":
    st.header("Model Prediction")
    
    # Option to upload an image or take a picture with the camera
    camera_image = st.camera_input("Take a picture with your camera:")
    uploaded_image = st.file_uploader("or Choose an Image:", type=["jpg", "jpeg", "png"])
    
    # Use the uploaded image or camera image for prediction
    if uploaded_image is not None:
        test_image = uploaded_image
    elif camera_image is not None:
        # Convert camera image to Pillow Image
        img_bytes = io.BytesIO(camera_image.read())
        image = Image.open(img_bytes)
        
        # Save the image as a temporary file
        image.save("temp.jpg")
        test_image = "temp.jpg"
    else:
        test_image = None

    if test_image is not None:
        st.image(test_image, width=400, use_container_width=True)

        # Predict button
        if st.button("Predict"):
            with st.spinner("Making a prediction..."):
                result_index, confidence = model_prediction(test_image, confidence_threshold=90)

                # If the confidence is sufficient
                if result_index is not None:
                    # Reading Labels
                    with open("labels.txt") as f:
                        content = f.readlines()
                    label = [i.strip() for i in content]
                    food_item = label[result_index]
                    
                    if food_item[0].lower() in 'aeiou':
                        st.success(f"Model predicts it's an **{food_item}** with confidence **{confidence * 100:.2f}%**")
                    else:
                        st.success(f"Model predicts it's a **{food_item}** with confidence **{confidence * 100:.2f}%**")

                    # Display nutrition information
                    nutrition_info = nutrition_data[nutrition_data['Food'] == food_item]
                    if not nutrition_info.empty:
                        st.subheader("Nutrition Information:")
                        st.table(nutrition_info)
                    else:
                        st.warning("Nutrition information not found for this item.")
                else:
                    st.warning(f"The model couldn't confidently predict the item. Confidence was only **{confidence * 100:.2f}%**.")
