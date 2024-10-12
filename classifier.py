'''
The Classifier class handles all image classification. It requires a .keras model file to load, and uses
prediction methods to process frames provided by the camera feed. Then, it returns a prediction result back to
the main application.
'''
from keras.models import load_model
from PIL import Image, ImageTk
import numpy as np


class Classifier:
    def __init__(self, model_path, labels_path):
        #Attempt to load model and extract labels
        try:
            self.model = load_model(model_path, compile = False)
            self.class_names = open(labels_path, "r").readlines()
            print("Model loaded.")
        except Exception as err:
            print("Error loading model:", err)
        

    def predict(self, image):
        '''
        Image Dimensions:
            A single 128x128 image has 16,384 pixels.
            Each pixel has 3 values (one for each color channel: Red, Green, Blue).
        Reshape Process:
            When we reshape the image to (1, 128, 128, 3), we’re essentially adding a batch dimension.
            This means we have one image of size 128x128 with 3 color channels.
            The total number of elements is 1 * 128 * 128 * 3 = 49,152.
        Resizing and Reshaping:
            First, we ensure the image is resized to 128x128 pixels.
            Then, convert it to a numpy array and reshape it to include the batch dimension and color channels.
        '''
        # 'image' is a PhotoImage object
        image = ImageTk.getimage(image)
        image = image.convert('RGB')  #Ensure that the image has 3 channels (RGB)
        image = image.resize((128, 128))  # Resize to 128x128
        
        # Make the image a numpy array and reshape it to the models input shape.
        image = np.asarray(image, dtype=np.float32).reshape(1, 128, 128, 3)
        # Normalize the image array
        image = (image / 127.5) - 1

        #Prediction
        prediction = self.model.predict(image)
        index = np.argmax(prediction)
        class_name = self.class_names[index]
        confidence_score = prediction[0][index]

        # Print prediction and confidence score
        #print("Class:", class_name, end="")
        #print("Confidence Score:", str(np.round(confidence_score * 100))[:-2], "%")
        output = (class_name, str(np.round(confidence_score * 100))[:-2])
        
        return output



