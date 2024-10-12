'''
Handles the CameraFeed class, which captures video frames and feeds them to the application.
'''
import cv2
from PIL import Image, ImageTk

class CameraFeed:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)

    def get_current_frame(self):
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame)
            imgtk = ImageTk.PhotoImage(image=img)
            
        return imgtk

    def __del__(self):
        if self.cap.isOpened():
            self.cap.release()