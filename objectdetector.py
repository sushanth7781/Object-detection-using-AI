import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
import requests
from io import BytesIO
import matplotlib.pyplot as plt

# Load the pre-trained MobileNetV2 model with ImageNet weights
model = MobileNetV2(weights='imagenet')

def label_image_from_url(image_url):
    # Fetch the image from the provided URL
    response = requests.get(image_url)
    img = cv2.imdecode(np.frombuffer(response.content, np.uint8), -1)
    
    if img is None:
        print("Error: Unable to load image from the provided URL.")
        return
    
    # Convert the image from BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Resize the image to 224x224 pixels
    img = cv2.resize(img, (224, 224))
    
    # Expand dimensions to match the expected input shape for MobileNetV2
    img_array = np.expand_dims(img, axis=0)
    
    # Preprocess the image array
    img_array = preprocess_input(img_array)
    
    # Make predictions using the pre-trained model
    predictions = model.predict(img_array)
    
    # Decode the predictions
    decoded_predictions = decode_predictions(predictions)
    
    # Get the label and confidence of the top prediction
    label = decoded_predictions[0][0][1]
    confidence = decoded_predictions[0][0][2]
    
    # Print the label and confidence
    print(f"Label: {label}, Confidence: {confidence:.2f}")
    
    # Display the image with the label and confidence as the title
    plt.imshow(img)
    plt.title(f"{label} ({confidence:.2f})")
    plt.axis('off')
    plt.show()

# Image URL for testing
image_url = 'https://hips.hearstapps.com/hmg-prod/images/2025-ford-mustang-60th-anniversary-exterior-66227932bb88e.jpg'
# Label the image from the provided URL
label_image_from_url(image_url)