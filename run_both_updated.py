import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import streamlit as st
import time
import dlib
from PIL import Image

# Load the trained model
model = load_model('bmi_model_mod.h5')

# Load the face detector
detector = dlib.get_frontal_face_detector()

# Define a function to preprocess the input image
def preprocess_image(image):
    image = image.resize((224, 224))
    image = np.array(image) / 255.0  # Normalize the image
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Create a function to predict BMI from an uploaded image
def predict_bmi_from_image(image):
    processed_image = preprocess_image(image)
    bmi_prediction = model.predict(processed_image)
    return bmi_prediction[0][0]

# Create a function to capture live video from the camera and predict BMI
def predict_bmi_live(frame):
    # Preprocess the frame
    processed_image = preprocess_image(frame)
    
    # Convert the frame to grayscale for face detection
    gray = np.array(frame.convert('L'))
    
    # Perform face detection
    faces = detector(gray)
    
    # Iterate over detected faces
    for face in faces:
        # Extract the face region from the frame
        x, y, w, h = face.left(), face.top(), face.width(), face.height()
        face_image = frame.crop((x, y, x + w, y + h))
        
        # Preprocess the face image
        processed_face = preprocess_image(face_image)
        
        # Predict BMI
        bmi_prediction = model.predict(processed_face)
        
        # Display the predicted BMI on the frame
        bmi_text = "Predicted BMI: {:.2f}".format(bmi_prediction[0][0]).zfill(5)
        draw = ImageDraw.Draw(frame)
        draw.text((x, y - 10), bmi_text, fill=(0, 255, 0))
    
    st.image(frame, channels='RGB')

# Create a Streamlit app
def main():
    st.title("BMI Prediction")

    # Display the options for input
    input_option = st.selectbox("Choose an option:", ("Webcam Input", "Upload Image"))

    if input_option == "Webcam Input":
        st.write("Please wait while the webcam stream is loading...")
        cap = cv2.VideoCapture(0)  # Open the camera
        time.sleep(1)
        ret, frame = cap.read()  # Read a frame from the camera     
        predict_bmi_live(frame)
        
    elif input_option == "Upload Image":
        uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            predict_bmi_from_image(image)
        else:
            st.write("No image uploaded.")

# Run the Streamlit app
if __name__ == "__main__":
    main()
