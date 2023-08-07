import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.preprocessing.image import ImageDataGenerator
from PIL import Image
import tkinter as tk
from tkinter import filedialog
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import os


class App():
    def __init__(self):
        self.createGUI()

    def createGUI(self):
        self.root = tk.Tk()
        self.root.title("Emotion Recognition")

        # Create drop-down list to select model name
        self.model_var = tk.StringVar()
        self.model_var.set("cnnModel.h5")  # Set default model
        self.model_options = ["cnnModel.h5",
        "svm_model.joblib" ]  # List of available models
        self.model_dropdown = tk.OptionMenu(self.root, self.model_var, *self.model_options)
        self.model_dropdown.pack()

        # Create button to select image file
        self.select_btn = tk.Button(self.root, text="Select Image", command=self.selectImage)
        self.select_btn.pack()

        # Create matplotlib figure to display image
        self.fig, self.ax = plt.subplots()
        self.ax.axis('off')
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().pack()

        # Create label to display predicted emotion
        self.emotion_label = tk.Label(self.root, text="")
        self.emotion_label.pack()

        self.root.mainloop()
    
    
    def selectImage(self):
        # Open file dialog box to select image file
        file_path = filedialog.askopenfilename()
        if os.path.isfile(file_path):
            try:
                test_img = cv2.imread(file_path, cv2.IMREAD_ANYCOLOR)
                if test_img is not None:
                    # Load pre-trained model
                    model_name = self.model_var.get()  # Get selected model name from drop-down list
                    
                    
                    import joblib
                    model_file = "models/" + model_name
                    
                    if os.path.isfile(model_file):
                        ext = os.path.splitext(model_file)[1]
                        if ext == ".h5":
                            # Load Keras model
                          self.Emotionrecognition(file_path)
                        elif ext == ".joblib":
                          self.Emotionrecognition_svm(file_path)
                        else:
                            print("Invalid model file format")
                    else:
                        print("Model file not found")
                else:
                    raise ValueError("Cannot read image file")
            except Exception as e:
                print(e)
        else:
            print("Invalid file path")


    def Emotionrecognition(self, imagepath):

        # Define haar cascade classifier for face detection
        face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
        test_img = cv2.imread(imagepath)
        img_bcp = test_img.copy()
    
        # Convert image to gray scale OpenCV
        gray_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
    
        # Detect face using haar cascade classifier
        faces_coordinates = face_classifier.detectMultiScale(gray_img)
    
        # Draw a rectangle around the faces
        for (x, y, w, h) in faces_coordinates:
            # Draw rectangle around face
            cv2.rectangle(test_img, (x, y), (x + w, y + h), (255, 0, 0), 2)
    
            # Crop face from image
            cropped_face = img_bcp[y:y+h, x:x+w]
    
            # Plot original image
            self.ax.clear()
            self.ax.imshow(cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB))
            self.canvas.draw()
    
            # Creating class dictionary
            class_dictionary = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}
    
            final_image = cv2.resize(cropped_face, (128, 128))
            final_image = np.expand_dims(final_image, axis=0)  # Need 4th dimension
            final_image = final_image/128.0  # Normalizing
            
            
            
            # Load pre-trained model
            model_name = self.model_var.get()  # Get selected model name from drop-down list
            
            import os
            import joblib
            model_file = "models/" + model_name
            
            if os.path.isfile(model_file):
                ext = os.path.splitext(model_file)[1]
                if ext == ".h5":
                    # Load Keras model
                    model = tf.keras.models.load_model(model_file)
                elif ext == ".joblib":
                    # Load joblib model
                    model = joblib.load(model_file)
                else:
                    print("Invalid model file format")
            else:
                print("Model file not found")
    
            # Make prediction on cropped face image
            prediction = model.predict(final_image)
            predicted_class = np.argmax(prediction)
    
            # Get predicted emotion label from class dictionary
            predicted_emotion = class_dictionary[predicted_class]
    
            # Show predicted emotion on the face in the image
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(test_img, predicted_emotion, (x, y-10), font,1, (0, 255, 0), 2)
    
            # Update emotion label on GUI
            self.emotion_label.config(text=predicted_emotion)
    
        # Plot image with predicted emotion on faces
        self.ax.clear()
        self.ax.imshow(cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB))
        self.canvas.draw()

    def Emotionrecognition_svm(self, imagepath):
        import tkinter as tk
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        import matplotlib.pyplot as plt
        import cv2
        import joblib
        # Load image and detect faces
        img_path = imagepath
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
            loaded_model = joblib.load("models/svm_model.joblib")

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
        
               
app = App()