import streamlit as st
import pandas as pd
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import cv2

model = load_model('bmi_model_mod.h5')

def predict_bmi(image):
    # Preprocess the image
    img = image.resize((224, 224))  # Resize the image to match the input size of your model
    img_array = np.array(img)
    img_array = img_array / 255.0  # Normalize the pixel values

    # Expand dimensions to create a batch of size 1
    img_array = np.expand_dims(img_array, axis=0)

    # Make the prediction
    prediction = model.predict(img_array)
    bmi = prediction[0][0]  # Assuming your model returns a single scalar value

    return bmi


def main():
    st.title("BMI Prediction App")

    # Use the camera as input
    cap = cv2.VideoCapture(0)

    if st.button("Capture Image"):
        _, frame = cap.read()

        # Convert the image array to PIL image
        image = Image.fromarray(frame)

        # Display the captured image
        st.image(image, caption="Captured Image", use_column_width=True)

        # Predict BMI
        bmi = predict_bmi(image)

        st.write("Predicted BMI:", bmi)

    # Release the camera
    cap.release()

if __name__ == "__main__":
    main()
