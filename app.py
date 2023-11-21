import streamlit as st
import tensorflow as tf
import cv2
import numpy as np


# Load your trained model
model = load_model(r)

# Load your trained model
model = tf.keras.models.load_model("C:/Users/Aditya/Downloads/Project/dog_cat_classification/model/saved_model.pb")


# Load your trained model
# model = tf.keras.models.load_model("model/saved_model.pb")

def load_and_prep_image(uploaded_file):
    # Read in the image
    bytes_data = uploaded_file.getvalue()
    image = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

    # Resize and normalize the image
    image = cv2.resize(image, (180, 180))  # Adjust the size to match your model's input
    image = image / 255.0  # Normalize pixel values

    # Reshape the image to fit the model input
    image = np.reshape(image, (1, 180, 180, 3))
    return image


st.title("Cat and Dog Image Classifier")

# User upload
uploaded_file = st.file_uploader("Upload an image of a cat or dog", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    # Display the uploaded image
    st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)
    st.write("Classifying...")

    # Process the uploaded image
    image = load_and_prep_image(uploaded_file)

    # Make prediction
    prediction = model.predict(image)
    
    # Display results
    if prediction[0] > 0.5:
        st.write("It's a Dog 🐶")
    else:
        st.write("It's a Cat 😺")
