import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import streamlit as st
import time

# Load the pre-trained face detection model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load the trained model
try:
    model = load_model('bmi_model_mod.h5')
except:
    st.error("Failed to load the model file.")

# Define a function to preprocess the input image
def preprocess_image(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB format
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
    processed_frame = frame.copy()
    
    # Preprocess the frame
    processed_image = preprocess_image(frame)
    
    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Perform face detection
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    # Iterate over detected faces
    for (x, y, w, h) in faces:
        # Draw a bounding box around the face
        cv2.rectangle(processed_frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        
        # Extract the face region from the frame
        face_image = frame[y:y + h, x:x + w]
        
        # Preprocess the face image
        processed_face = preprocess_image(face_image)
        
        # Predict BMI
        bmi_prediction = model.predict(processed_face)
        
        # Display the predicted BMI on the frame
        bmi_text = "Predicted BMI: {:.2f}".format(bmi_prediction[0][0]).zfill(5)
        cv2.putText(processed_frame, bmi_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    return processed_frame

# Create a Streamlit app
def main():
    st.title("BMI Prediction")

    # Display the options for input
    input_option = st.selectbox("Choose an option:", ("Webcam Input", "Upload Image"))

    if input_option == "Webcam Input":
        st.write("Please wait while the webcam stream is loading...")
        cap = cv2.VideoCapture(0)  # Open the camera
        time.sleep(1)
        
        # Placeholder for displaying the processed frame
        processed_frame_placeholder = st.empty()
        
        while True:
            ret, frame = cap.read()  # Read a frame from the camera
            processed_frame = predict_bmi_live(frame)
            processed_frame_placeholder.image(processed_frame, channels="BGR")
            
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
