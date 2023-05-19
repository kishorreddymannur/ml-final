import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import streamlit as st
import time

# Load the pre-trained face detection model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load the trained BMI prediction model
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

# Create a function to capture live video from the camera and predict BMI
def predict_bmi_live(frame):
    processed_image = preprocess_image(frame)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        face_image = frame[y:y + h, x:x + w]
        processed_face = preprocess_image(face_image)
        bmi_prediction = model.predict(processed_face)
        bmi_text = "Predicted BMI: {:.2f}".format(bmi_prediction[0][0]).zfill(5)
        cv2.putText(frame, bmi_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    return frame

# Create a Streamlit app
def main():
    st.title("Face Detection and BMI Prediction")

    # Display the options for input
    input_option = st.selectbox("Choose an option:", ("Webcam Face Detection", "Upload Image for BMI Prediction"))

    if input_option == "Webcam Face Detection":
        st.write("Please wait while the webcam stream is loading...")
        cap = cv2.VideoCapture(0)  # Open the camera
        time.sleep(1)
        ret, frame = cap.read()  # Read a frame from the camera

        if ret:
            processed_frame = predict_bmi_live(frame)
            st.image(processed_frame, channels="BGR")
        else:
            st.write("Error capturing frame from webcam.")

    elif input_option == "Upload Image for BMI Prediction":
        uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

        if uploaded_file is not None:
            image = np.array(cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), 1))
            processed_image = preprocess_image(image)
            bmi_prediction = model.predict(processed_image)
            bmi_text = "Predicted BMI: {:.2f}".format(bmi_prediction[0][0]).zfill(5)
            cv2.putText(image, bmi_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            st.image(image, channels="BGR")
        else:
            st.write("No image uploaded.")

# Run the Streamlit app
if __name__ == "__main__":
    main()
