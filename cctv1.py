from tkinter import messagebox, filedialog, Text, Label, Button
import tkinter as tk
import numpy as np
import cv2
from ultralytics import YOLO
from collections import defaultdict
import uuid

# Global Variables
global filename, model, unique_ids, total_count
labels = ['Person', 'Crowd']
CONFIDENCE_THRESHOLD = 0.3
GREEN = (0, 255, 0)
unique_ids = set()
total_count = 0

def loadModel():
    global model
    text.delete('1.0', tk.END)
    model = YOLO("yolov8_model/best.pt")
    text.insert(tk.END, "YoloV8 Model Loaded")

def track_objects(detections, frame, is_video=False):
    global unique_ids, total_count
    count = 0
    current_ids = set()
    
    for data in detections.boxes.data.tolist():
        confidence, cls_id = data[4], int(data[5])
        if confidence < CONFIDENCE_THRESHOLD or cls_id != 0:
            continue
        xmin, ymin, xmax, ymax = map(int, data[:4])
        
        matched_id = str(uuid.uuid4())  # Assign a new unique ID
        unique_ids.add(matched_id)
        current_ids.add(matched_id)
        count += 1
        
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), GREEN, 2)
        cv2.putText(frame, str(count), (xmin, ymin - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
    
    if not is_video:
        total_count = count  # Ensure correct total count for images
    else:
        total_count = len(unique_ids)
    
    text.insert(tk.END, f"Total Count: {total_count}\n")
    
    if total_count > 100:
        text.insert(tk.END, "Overcrowding Alert! More than 100 people detected.\n")
        messagebox.showwarning("Overcrowding Alert", "More than 100 people detected!")

def imageDetection():
    global model, total_count, unique_ids
    unique_ids.clear()
    total_count = 0
    text.delete('1.0', tk.END)
    filename = filedialog.askopenfilename(initialdir="images")
    text.insert(tk.END, filename + " loaded\n\n")
    frame = cv2.imread(filename)
    frame = cv2.resize(frame, (700, 600))
    detections = model(frame)[0]
    track_objects(detections, frame, is_video=False)
    cv2.imshow("Crowd Detection from Image", frame)
    cv2.waitKey(0)

def videoDetection():
    global model, unique_ids, total_count
    unique_ids.clear()
    total_count = 0
    text.delete('1.0', tk.END)
    filename = filedialog.askopenfilename(initialdir="Video")
    text.insert(tk.END, filename + " loaded\n\n")
    video_cap = cv2.VideoCapture(filename)
    
    while True:
        ret, frame = video_cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (900, 800))  # Increased frame size
        detections = model(frame)[0]
        track_objects(detections, frame, is_video=True)
        cv2.imshow("Crowd Detection from Video", frame)
        if cv2.waitKey(50) == ord("q"):
            break
    
    video_cap.release()
    cv2.destroyAllWindows()

def graph():
    graph_img = cv2.imread('yolov8_model/results.png')
    graph_img = cv2.resize(graph_img, (800, 600))
    cv2.imshow("YoloV8 Training Graph", graph_img)
    cv2.waitKey(0)

def close():
    main.destroy()

main = tk.Tk()
main.title("Using Existing CCTV Network for Crowd Management")
main.geometry("1300x1200")

font = ('times', 16, 'bold')
title = Label(main, text='Using Existing CCTV Network for Crowd Management')
title.config(bg='greenyellow', fg='dodger blue', font=font, height=3, width=120)
title.place(x=0, y=5)

font1 = ('times', 12, 'bold')
text = Text(main, height=20, width=150)
text.place(x=50, y=120)
text.config(font=font1)

font2 = ('times', 13, 'bold')
loadButton = Button(main, text="Generate & Load YoloV8 Model", command=loadModel, font=font2)
loadButton.place(x=50, y=550)

imageButton = Button(main, text="Crowd Management from Images", command=imageDetection, font=font2)
imageButton.place(x=330, y=550)

videoButton = Button(main, text="Crowd Management from Videos", command=videoDetection, font=font2)
videoButton.place(x=630, y=550)

graphButton = Button(main, text="YoloV8 Training Graph", command=graph, font=font2)
graphButton.place(x=50, y=600)

exitButton = Button(main, text="Exit", command=close, font=font2)
exitButton.place(x=330, y=600)

main.config(bg='LightSkyBlue')
main.mainloop()