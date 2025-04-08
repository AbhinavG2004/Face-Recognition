import cv2
import numpy as np
import torch
import tkinter as tk
from tkinter import Label, simpledialog
from PIL import Image, ImageTk
from torchvision import models, transforms
from collections import deque
import os
import datetime

# === Load pretrained ResNet18 model ===
resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
resnet.fc = torch.nn.Identity()
resnet.eval()

# === Face detection using OpenCV ===
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# === Preprocessing for face ===
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# === Database of embeddings ===
known_embeddings = {}
smoothing_window = 10
smoothing_queue = deque(maxlen=smoothing_window)

# === Load known embeddings from folder structure ===
def load_known_faces(folder=os.path.join("Dataset", "Dataset", "Original Images", "Original Images")):
    for person in os.listdir(folder):
        person_folder = os.path.join(folder, person)
        if not os.path.isdir(person_folder):
            continue
        embeddings = []
        for img_name in os.listdir(person_folder):
            img_path = os.path.join(person_folder, img_name)
            if not img_path.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            img = cv2.imread(img_path)
            if img is None:
                continue
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            for (x, y, w, h) in faces:
                face = img[y:y + h, x:x + w]
                if face.size == 0:
                    continue
                face_tensor = transform(face).unsqueeze(0)
                with torch.no_grad():
                    embedding = resnet(face_tensor).squeeze().numpy()
                embeddings.append(embedding)
        if embeddings:
            known_embeddings[person] = np.stack(embeddings)

load_known_faces()

# === Compare embeddings with threshold ===
def recognize_face(face_img, threshold=0.6):
    if face_img.size == 0:
        return "Unknown"
    face_tensor = transform(face_img).unsqueeze(0)
    with torch.no_grad():
        embedding = resnet(face_tensor).squeeze().numpy()

    min_dist = float("inf")
    best_match = "Unknown"

    for name, embeddings in known_embeddings.items():
        dists = np.linalg.norm(embeddings - embedding, axis=1)
        mean_dist = np.mean(dists)
        if mean_dist < min_dist:
            min_dist = mean_dist
            best_match = name

    if min_dist < threshold:
        return best_match
    else:
        return "Unknown"

# === Try different camera indices ===
def get_camera():
    for i in range(3):  # try camera 0, 1, 2
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            print(f"✅ Webcam opened on index {i}")
            return cap
        cap.release()
    print("❌ ERROR: Cannot open any webcam.")
    return None

# === GUI Setup ===
window = tk.Tk()
window.title("Live Face Recognition")
window.attributes("-fullscreen", True)

label = Label(window)
label.pack()

cap = get_camera()
if cap is None:
    window.destroy()
    exit()

# === Save new face to dataset ===
def save_new_face(face_img, person_name):
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    save_path = os.path.join("Dataset", "Dataset", "Original Images", "Original Images", person_name)
    os.makedirs(save_path, exist_ok=True)
    filename = f"{person_name}_{timestamp}.jpg"
    cv2.imwrite(os.path.join(save_path, filename), face_img)
    print(f"✅ Saved new face to {save_path}")
    load_known_faces()  # Reload embeddings

# === Label unknown face ===
def label_unknown(face_img):
    window.update()
    person_name = simpledialog.askstring("New Face Detected", "Please enter your name:")
    if person_name:
        for _ in range(5):
            ret, frame = cap.read()
            if not ret:
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            for (x, y, w, h) in faces:
                face = frame[y:y + h, x:x + w]
                if face.size == 0:
                    continue
                save_new_face(face, person_name)
                break

# === Update GUI frames ===
def update_frame():
    ret, frame = cap.read()
    if not ret or frame is None:
        print("❌ ERROR: Failed to read from camera.")
        label.after(1000, update_frame)
        return

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    predictions = []

    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]
        if face.size == 0:
            continue
        name = recognize_face(face)
        predictions.append(name)
        if name == "Unknown":
            label_unknown(face)
            return  # Restart after labeling

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, name, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

    # === Smoothing predictions ===
    if predictions:
        smoothing_queue.append(predictions[0])
        most_common = max(set(smoothing_queue), key=smoothing_queue.count)
        cv2.putText(frame, f"Smoothed: {most_common}", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

    # === Convert frame for Tkinter ===
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(rgb)
    imgtk = ImageTk.PhotoImage(image=img)
    label.imgtk = imgtk
    label.configure(image=imgtk)

    label.after(10, update_frame)

# === Exit app on Escape key ===
def on_key(event):
    if event.keysym == 'Escape':
        print("👋 Exiting...")
        cap.release()
        window.destroy()
        cv2.destroyAllWindows()

window.bind("<Key>", on_key)

# === Start frame loop ===
update_frame()
window.mainloop()

# === Cleanup on close ===
cap.release()
cv2.destroyAllWindows()
