import streamlit as st
import torch
import numpy as np
import cv2
from PIL import Image
import pyttsx3

# Initialize the text-to-speech engine
tts_engine = pyttsx3.init()

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

def detect_objects(image):
    # Convert the image to RGB
    img_rgb = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)
    results = model(img_rgb)

    # Render the results on the image
    img_rgb_with_boxes = results.render()[0]
    img_rgb_with_boxes = cv2.cvtColor(img_rgb_with_boxes, cv2.COLOR_BGR2RGB)

    # Extract detections and confidence scores
    detections = []
    for pred in results.xyxy[0]:  # results.xyxy[0] contains detections
        label = model.names[int(pred[-1])]
        confidence = float(pred[4])
        detections.append((label, confidence))

    return Image.fromarray(img_rgb_with_boxes), detections

def speak_labels(detections):
    labels = [label for label, _ in detections]
    if labels:
        labels_text = ', '.join(labels)
        tts_engine.say(f"Detected objects are: {labels_text}")
        tts_engine.runAndWait()
    else:
        tts_engine.say("No objects detected.")
        tts_engine.runAndWait()

def main():
    st.title("Object Detection using YOLOv5 with TTS")
    
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        st.write("")
        st.write("Detecting objects...")

        result_image, detections = detect_objects(image)
        st.image(result_image, caption='Processed Image', use_column_width=True)
        
        if detections:
            st.write("Detected objects:")
            for label, confidence in detections:
                st.write(f"Label: {label}, Confidence: {confidence:.2f}")
            # Call the speak_labels function to announce the detected objects
            speak_labels(detections)
        else:
            st.write("No objects detected.")
            # Announce that no objects were detected
            tts_engine.say("No objects detected.")
            tts_engine.runAndWait()

if __name__ == "__main__":
    main()
