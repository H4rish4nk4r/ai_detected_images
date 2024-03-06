import os
import psycopg2
import numpy as np
from PIL import Image
from imgbeddings import imgbeddings
import cv2

# Function to calculate embeddings for an image
def calculate_embedding(image):
    ibed = imgbeddings()
    return ibed.to_embeddings(image)

# Function to compare embeddings
def compare_embeddings(embedding1, embedding2):
    return np.linalg.norm(embedding1 - embedding2)

# Function to find matching face in the stored-faces directory
def find_matching_face(embedding):
    min_distance = float('inf')
    matching_face = None

    for filename in os.listdir("stored-faces"):
        if filename.endswith(".jpg"):
            stored_face_path = os.path.join("stored-faces", filename)
            stored_face = Image.open(stored_face_path)
            stored_embedding = calculate_embedding(stored_face)
            distance = compare_embeddings(embedding, stored_embedding)
            if distance < min_distance:
                min_distance = distance
                matching_face = stored_face_path
    
    return matching_face

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Capture an image from the webcam
ret, frame = cap.read()

# Release the webcam
cap.release()

# Convert the captured frame to RGB format
frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

# Convert the frame to a PIL Image
img = Image.fromarray(frame_rgb)

# Calculate the embeddings for the webcam-captured image
webcam_embedding = calculate_embedding(img)

# Find the matching face in the stored-faces directory
matching_face_path = find_matching_face(webcam_embedding)

# If a matching face is found, save the webcam-captured image
if matching_face_path:
    img.save("matched_face.jpg")
    print("Webcam-captured image matched with a face in stored-faces directory and saved as matched_face.jpg")
else:
    print("Webcam-captured image did not match with any faces in stored-faces directory")
