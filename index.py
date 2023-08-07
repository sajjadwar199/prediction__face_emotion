import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.preprocessing.image import ImageDataGenerator
from PIL import Image

class App():
   

    def __init__(self):
          # self.trainModel()
          self.Emotionrecognition("testingImages/test7.jpg")
         # self.check_image_upload_size("testingImages/v.png")
     # يأتي بلبيانات على هيئة           

    #  كشف الوجه وقصه وكشف الأيماءات

    def Emotionrecognition(self, imagepath):

        # Define haar cascade classifier for face detection
        face_classifier = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        test_img = cv2.imread(imagepath)
        # test_img= cv2.resize(test_img,)
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
            plt.subplot(1, 2, 1)
            plt.imshow(cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB))

        # Plot cropped image after performing face detection
            # plt.subplot(1, 2, 2)
            # cutFaceimage = plt.imshow(cv2.cvtColor(
            #  cropped_face, cv2.COLOR_BGR2RGB))
            #  Creating class dictionary
            class_dictionary = {0: 'angry', 1: 'disgust', 2: 'fear',
                                3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}

            final_image = cv2.resize(cropped_face, (128, 128))
            final_image = np.expand_dims(
                final_image, axis=0)  # Need 4th dimension
            final_image = final_image/128.0  # Normalizing

            # Load model
            new_model = tf.keras.models.load_model('models/CnnModel.h5')

            prediction = new_model.predict(final_image)
            print(np.argmax(prediction))
            print(class_dictionary[np.argmax(prediction)])
            # Convert 4-d image to 3d
            img_3d = np.squeeze(img_bcp)

            # Define opencv font style
            font = cv2.FONT_HERSHEY_PLAIN

            # Draw a rectangle around the faces
            for (x, y, w, h) in faces_coordinates:
                # Draw rectangle around face
                cv2.rectangle(img_3d, (x, y), (x + w, y + h), (0, 128, 0), 2)

            # Write face emotion class text on image
            cv2.putText(img_3d, class_dictionary[np.argmax(
                prediction)], (70, 70), font, 2, (0, 0, 270), 2, cv2.LINE_4)
            print(prediction)
            print(class_dictionary[np.argmax(prediction)])
            import tkinter as tk
            from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
            cv2.cvtColor(img_3d, cv2.COLOR_BGR2RGB)
            img_rgb= cv2.cvtColor(img_3d, cv2.COLOR_BGR2RGB)
            fig, ax = plt.subplots()
            ax.imshow(img_rgb)
            
            # create a Tkinter window
            root = tk.Tk()
            
            # add the plot to the Tkinter window
            canvas = FigureCanvasTkAgg(fig, master=root)
            canvas.draw()
            canvas.get_tk_widget().pack()
            
            # display the Tkinter window
            tk.mainloop()
             





            # Show output image
            plt.imshow(cv2.cvtColor(img_3d, cv2.COLOR_BGR2RGB))
        return cropped_face

    def trainModel(self):

        # Define the CNN model architecture
        model = Sequential()
        model.add(Conv2D(32, (3, 3), activation='relu',
                  input_shape=(128, 128, 3)))
        model.add(MaxPooling2D((2, 2)))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D((2, 2)))
        model.add(Conv2D(128, (3, 3), activation='relu'))
        model.add(MaxPooling2D((2, 2)))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(7, activation='softmax'))

        # Compile the model
        model.compile(optimizer='adam',
                      loss='categorical_crossentropy', metrics=['accuracy'])

        # Load and preprocess the data
        train_datagen = ImageDataGenerator(
            rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
        test_datagen = ImageDataGenerator(rescale=1./255)

        train_generator = train_datagen.flow_from_directory(
            'dataset/train/train', target_size=(128, 128), batch_size=32, class_mode='categorical')
        validation_generator = test_datagen.flow_from_directory(
            'dataset/test/test', target_size=(128, 128), batch_size=32, class_mode='categorical')

        # Train the model on the data
        model.fit(train_generator, epochs=25,
                  validation_data=validation_generator)

        # Save the model weights
        model.save_weights('model4_weights.h5')
        model.save('model4.h5')

        # Predict on a new image
 
            
        #لفحص حجم الصورة قبل رفعها 
    # def check_image_upload_size(self,file_path, max_size=800, max_width=800, max_height=128):
    #    from PIL import Image
    #    image=Image.open(file_path)
    #    # if (image.size[0] > max_width and image.size[1]>max_height):
    #    #  print("size must be 128 x 128")   
    #     # (width,height) tuple
    #    image.format # (keeps the image format)
    #    image.resize((800,800))
    #    return image 
app = App()
