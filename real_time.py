import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array

# Load the pre-trained model and label dictionary
model = tf.keras.models.load_model('models/CnnModel.h5')
class_dictionary = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}

# Create a window for displaying the video stream
cv2.namedWindow('Real-time Emotion Detection', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Real-time Emotion Detection', 640, 480)

# Initialize the video capture object
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Load the Haar Cascade Classifier for face detection
face_cascade = cv2.CascadeClassifier('C:\\Users\\sajjadpc\\AppData\\Local\\Programs\\Python\\Python310\\Lib\\site-packages\\cv2\\data\\haarcascade_frontalface_default.xml')
if face_cascade.empty():
    raise IOError('Unable to load the face cascade classifier XML file')

while True:
    # Capture a frame from the video stream
    ret, frame = cap.read()
    if not ret:
        break

    # Check if the frame size is valid
    print(frame.shape)

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)

    # Loop through the detected faces and perform emotion detection
    for (x, y, w, h) in faces:
        # Crop the face region
        roi = gray[y:y+h, x:x+w]

        # Preprocess the face region for prediction
        color_roi = cv2.cvtColor(roi, cv2.COLOR_GRAY2RGB)
        roi = cv2.resize(color_roi, (128, 128))
        roi = roi.astype("float") / 255.0
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)

        # Predict the emotion using the pre-trained model
        preds = model.predict(roi)[0]
        label = class_dictionary[preds.argmax()]

        # Draw a rectangle around the face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Put the label text on the frame
        cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('Real-time Emotion Detection', frame)

    # Exit if the 'q' key is pressed
    if cv2.waitKey(1) == ord('q'):
        break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()
