import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import streamlit as st
import time

# Load the trained model
model = load_model('bmi_model_mod.h5')

# Define a function to preprocess the input image
def preprocess_image(image):
    image = tf.image.resize(image, (224, 224))
    image = tf.cast(image, tf.float32) / 255.0  # Normalize the image
    image = tf.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Create a function to predict BMI from an uploaded image
def predict_bmi_from_image(image):
    processed_image = preprocess_image(image)
    bmi_prediction = model.predict(processed_image)
    return bmi_prediction[0][0]

# Create a Streamlit app
def main():
    st.title("BMI Prediction")

    # Display the options for input
    input_option = st.selectbox("Choose an option:", ("Webcam Input", "Upload Image"))

    if input_option == "Webcam Input":
        st.write("Please wait while the webcam stream is loading...")
        camera = st.camera()
        time.sleep(1)
        
        while True:
            resized_frame = np.array(Image.fromarray(camera).resize((224, 224)))
            predict_bmi_from_image(resized_frame)
            
            if st.button("Stop"):
                break
        
    elif input_option == "Upload Image":
        uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            image = np.array(Image.open(uploaded_file).resize((224, 224)))
            bmi_prediction = predict_bmi_from_image(image)
            st.markdown("<h3 style='text-align: center;'>Predicted BMI: {:.2f}</h3>".format(bmi_prediction), unsafe_allow_html=True)
            st.image(image, channels="RGB")
        else:
            st.write("No image uploaded.")

# Run the Streamlit app
if __name__ == "__main__":
    main()
