import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import streamlit as st
import time

# Load the pre-trained face detection model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load the trained model
model = load_model('bmi_model_mod.h5')

# Define a function to preprocess the input image
def preprocess_image(image):
    image = cv2.resize(image, (224, 224))
    image = image / 255.0  # Normalize the image
    image = np.expand_dims(image, axis=0)  # Add batch dimension
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
        cap = cv2.VideoCapture(0)  # Open the camera
        time.sleep(1)
        
        while True:
            ret, frame = cap.read()  # Read a frame from the camera
            predict_bmi_live(frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
    elif input_option == "Upload Image":
        uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            image = np.array(cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), 1))
            bmi_prediction = predict_bmi_from_image(image)
            st.markdown("<h3 style='text-align: center;'>Predicted BMI: {:.2f}</h3>".format(bmi_prediction), unsafe_allow_html=True)
            st.image(image, channels="BGR")
        else:
            st.write("No image uploaded.")

# Run the Streamlit app
if __name__ == "__main__":
    main()
