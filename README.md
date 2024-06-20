# Object-Detector-python
This script uses TensorFlow and OpenCV to label an image from a URL using the MobileNetV2 model pre-trained on the ImageNet dataset. The code fetches the image, preprocesses it, performs a prediction, and displays the image with the predicted label and confidence score.



1. Imports:<br>
cv2 for image processing.<br>
numpy for numerical operations.<br>
tensorflow for using the MobileNetV2 model.<br>
requests for fetching the image from the URL.<br>
matplotlib.pyplot for displaying the image.<br>
<br>
<br>

2. Model Loading:<br>
MobileNetV2 is loaded with weights pre-trained on the ImageNet dataset.<br>
<br>
<br>

3. Image Fetching and Preprocessing:<br>
The image is fetched from the URL using requests.<br>
The image is decoded using cv2.imdecode and converted from BGR to RGB format.<br>
The image is resized to 224x224 pixels, as required by MobileNetV2.<br>
The image array is expanded to match the input shape expected by the model.<br>
The image is preprocessed using preprocess_input.<br>
<br>
<br>

4. Prediction and Display:<br>
The model predicts the label of the image.<br>
The predictions are decoded using decode_predictions.<br>
The label and confidence of the top prediction are printed.<br>
The image is displayed using matplotlib.pyplot with the label and confidence shown in the title.