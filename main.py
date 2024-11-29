import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
from PIL import Image
import cv2
import numpy as np

# Load nutrition data
nutrition_data = pd.read_csv("nutrition_info.csv")  # Ensure this CSV contains food items and their nutrition info

# Tensorflow Model Prediction
def model_prediction(test_image):
    model = tf.keras.models.load_model("trained_model.h5")
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(64, 64))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])  # convert single image to batch
    predictions = model.predict(input_arr)
    return np.argmax(predictions)  # return index of max element

# Custom CSS for styling
st.markdown(
    """
    <style>
    body {
        background-image: url('https://example.com/your-background-image.jpg'); /* Replace with your background image URL */
        background-size: cover;
        color: blue;
    }
    .sidebar .sidebar-content {
        background-color: rgba(0, 0, 0, 0.7);
        border-radius: 10px;
        padding: 20px;
    }
    h1, h2, h3 {
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.7);
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
app_mode = st.sidebar.selectbox("Select Page", ["Home", "About Project", "Prediction"])

# Main Page
if app_mode == "Home":
    st.header("FRUITS & VEGETABLES RECOGNITION SYSTEM")
    image_path = "home_img.jpg"
    st.image(image_path, use_container_width=True)

# About Project
elif app_mode == "About Project":
    st.header("About Project")
    st.subheader("About Dataset")
    st.text("This dataset contains images of the following food items:")
    st.code("fruits- banana, apple, pear, grapes, orange, kiwi, watermelon, pomegranate, pineapple, mango.")
    st.code("vegetables- cucumber, carrot, capsicum, onion, potato, lemon, tomato, raddish, beetroot, cabbage, lettuce, spinach, soy bean, cauliflower, bell pepper, chilli pepper, turnip, corn, sweetcorn, sweet potato, paprika, jalepeño, ginger, garlic, peas, eggplant.")
    st.subheader("Content")
    st.text("This dataset contains three folders:")
    st.text("1. train (100 images each)")
    st.text("2. test (10 images each)")
    st.text("3. validation (10 images each)")

# Prediction Page
elif app_mode == "Prediction":
    st.header("Model Prediction")
    
    # Option to upload an image or take a picture with the camera
    camera_image = st.camera_input("Take a picture with your camera:")
    uploaded_image = st.file_uploader("or Choose an Image:", type=["jpg", "jpeg", "png"])
    #camera_image = st.camera_input("Or take a picture with your camera:")
    
    # Use the uploaded image or camera image for prediction
    if uploaded_image is not None:
        test_image = uploaded_image
    elif camera_image is not None:
        img_array = np.array(camera_image)
        cv2.imwrite('temp.jpg', cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR))
        test_image = 'temp.jpg'
    else:
        test_image = None

    if test_image is not None:
        st.image(test_image, width=400, use_container_width=True)

        # Predict button
        if st.button("Predict"):
            with st.spinner("Making a prediction..."):
                result_index = model_prediction(test_image)

                # Reading Labels
                with open("labels.txt") as f:
                    content = f.readlines()
                label = [i.strip() for i in content]
                food_item = label[result_index]
                st.success("Model is Predicting it's a {}".format(food_item))

                # Display nutrition information
                nutrition_info = nutrition_data[nutrition_data['Food'] == food_item]
                if not nutrition_info.empty:
                    st.subheader("Nutrition Information:")
                    #st.write(nutrition_info.to_dict(orient='records')[0])  # Display the first record as a dictionary
                    st.table(nutrition_info)
                else:
                    st.warning("Nutrition information not found for this item.")
        # else:
        #     st.warning("Please choose an image file.qqq")