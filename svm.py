import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import cv2


# # Load FER-2013 dataset
# data = pd.read_csv("dataset/fer2013.csv")

# # Extract pixels and emotions from dataset
# pixels = data["pixels"].tolist()
# emotions = pd.get_dummies(data["emotion"]).values

# # Convert pixels from strings to 2D arrays
# pixels = np.array([np.fromstring(pixel, dtype='int', sep=' ').reshape((48, 48)) for pixel in pixels])

# # Split dataset into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(pixels, emotions, test_size=0.3, random_state=42)

# # Flatten each image to a 1D array
# X_train = X_train.reshape(X_train.shape[0], 48 * 48).astype('float32')
# X_test = X_test.reshape(X_test.shape[0], 48 * 48).astype('float32')

# # Normalize pixel values to [0, 1]
# X_train /= 255.0
# X_test /= 255.0

# # Train SVM
# svm_model = SVC(kernel='linear', C=1.0, random_state=42)
# svm_model.fit(X_train, y_train.argmax(axis=1))

# # Evaluate SVM on testing set
# y_pred = svm_model.predict(X_test)
# acc = accuracy_score(y_test.argmax(axis=1), y_pred)
# print("Test accuracy:", acc)

import joblib

 

# # Save trained model
# joblib.dump(svm_model, "svm_model.joblib")

# Load saved model
loaded_model = joblib.load("models/svm_model.joblib")

# # Load new image and preprocess it
# img_path = "testingimages/test10.jpg"
# test_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
# test_img = cv2.resize(test_img, (48, 48))
# test_img = test_img.astype('float32') / 255.0
# test_img = test_img.reshape(1, 48 * 48)
# import cv2

# # Load image and detect faces
# img_path = "testingimages/test8.jpg"
# img = cv2.imread(img_path)
# face_cascade = cv2.CascadeClassifier('C:\\Users\\sajjadpc\\AppData\\Local\\Programs\\Python\\Python310\\Lib\\site-packages\\cv2\\data\\haarcascade_frontalface_default.xml')
# faces = face_cascade.detectMultiScale(img, 1.3, 5)

# # Crop face from image
# for (x, y, w, h) in faces:
#     face_img = img[y:y+h, x:x+w]
#     face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
#     face_img = cv2.resize(face_img, (48, 48))
#     face_img = face_img.astype('float32') / 255.0
#     face_img = face_img.reshape(1, 48 * 48)

# # Make prediction on new image
# prediction = loaded_model.predict(face_img)
# predicted_emotion = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral'][prediction[0]]
# print("Predicted emotion:", predicted_emotion)


import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import cv2

# Load image and detect faces
img_path = "testingimages/test10.jpg"
img = cv2.imread(img_path)
face_cascade = cv2.CascadeClassifier('C:\\Users\\sajjadpc\\AppData\\Local\\Programs\\Python\\Python310\\Lib\\site-packages\\cv2\\data\\haarcascade_frontalface_default.xml')
faces = face_cascade.detectMultiScale(img, 1.3, 5)

# Create figure and subplots
fig, ax = plt.subplots(figsize=(6, 6))

# Plot the image with detected faces
ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
for (x, y, w, h) in faces:
    ax.add_patch(plt.Rectangle((x, y), w, h, linewidth=1, edgecolor='r', facecolor='none'))

# Crop face from image
for (x, y, w, h) in faces:
    face_img = img[y:y+h, x:x+w]
    face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    face_img = cv2.resize(face_img, (48, 48))
    face_img = face_img.astype('float32') / 255.0
    face_img = face_img.reshape(1, 48 * 48)

    # Make prediction on new image
    prediction = loaded_model.predict(face_img)
    predicted_emotion = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral'][prediction[0]]
    print("Predicted emotion:", predicted_emotion)

    # Display predicted emotion label on the plot
    text_x, text_y = x - 10, y - 10
    ax.text(text_x, text_y, predicted_emotion, fontsize=12, color='green')

# Set plot properties and create Tkinter window
ax.axis('off')
plt.tight_layout()
root = tk.Tk()
root.title("Emotion Detection")
root.geometry("800x600")

# Create a canvas to display the plot
canvas = FigureCanvasTkAgg(fig, master=root)
canvas.draw()
canvas.get_tk_widget().pack()

# Start the Tkinter event loop
tk.mainloop()


