'''
Handles the Application class, which executes the tkinter GUI program that displays frames
taken from the CameraFeed class.
'''

import tkinter as tk
from tkinter import Label
import cv2
from PIL import Image, ImageTk
from camera_feed import CameraFeed
from classifier import Classifier

class Application:
    def __init__(self, root):
        self.root = root
        self.root.title("Camera Feed")
        self.root.geometry("800x600")

        self.label = Label(self.root)
        self.label.pack()

        self.cap = CameraFeed()
        self.classifier = Classifier("shape_classifier_model.keras", "labels.txt")
        
        # Label to display the prediction
        self.label_pred = tk.Label(root, text="Prediction: ", font=("Arial", 16))
        self.label_pred.pack()

        # Label to display the confidence score
        self.label_conf = tk.Label(root, text="Confidence: ", font=("Arial", 16))
        self.label_conf.pack()

        self.update()

    def update(self):
        frame = self.cap.get_current_frame()
        try:
            prediction = self.classifier.predict(frame)
            class_name, confidence_score = prediction
            # Display the prediction and confidence on the GUI
            self.label_pred.config(text=f"Prediction: {class_name}")
            self.label_conf.config(text=f"Confidence: {confidence_score}%")
        except Exception as err:
            print(err)
    
        self.label.imgtk = frame
        self.label.configure(image=frame)
        self.root.after(10, self.update)


    def __del__(self):
        pass
        #self.cap.release()