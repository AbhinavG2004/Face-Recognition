import tkinter as tk
from PIL import Image, ImageTk
import cv2
import os
import time
import threading
from model import enhance_image, upscale_if_needed, segment_face, preprocess, model, match_face, detector, device
import torch
from datetime import datetime   
# Path to save/load known faces
known_faces_path = os.path.join("Dataset", "Faces", "Faces.pt")
# Load known faces from the correct path
if os.path.exists(known_faces_path):
    known_faces = torch.load(known_faces_path)
    print(f"[INFO] Loaded {len(known_faces)} known faces from {known_faces_path}")
else:
    known_faces = {}
    print("[INFO] No known_faces.pt found. Starting with empty dictionary.")
class FaceRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Live Face Recognition")
        self.video_label = tk.Label(root)
        self.video_label.pack()
        self.name_label = tk.Label(root, text="")
        self.name_label.pack()
        self.cap = None
        self.running = False
        self.last_predictions = []
        self.last_update_time = 0
        self.capture_stage = 0
        self.captured_angles = []
        self.capture_start_time = None
        self.capturing_unknown = False
        self.prompted_for_name = False
        self.name_entry = None
        self.submit_btn = None
    def start_recognition(self):
        self.cap = cv2.VideoCapture(0)
        self.running = True
        self.last_update_time = time.time()
        self.update_frame()
    def stop_recognition(self):
        self.running = False
        if self.cap:
            self.cap.release()
        self.cap = None
    def prompt_for_name(self):
        if not self.prompted_for_name:
            self.prompted_for_name = True
            self.name_label.config(text="Enter Name for Unknown Face:")
            self.name_entry = tk.Entry(self.root)
            self.name_entry.pack()
            self.submit_btn = tk.Button(self.root, text="Submit", command=self.save_unknown_face)
            self.submit_btn.pack()
    def save_unknown_face(self):
        name = self.name_entry.get().strip()
        if name:
            embeddings = []
            for idx, img in enumerate(self.captured_angles):
                filename = f"{name}{idx+1}.jpg"
                path = os.path.join("Dataset", "Faces", filename)
                cv2.imwrite(path, img)
                rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(rgb_img)
                segmented_np = segment_face(pil_img)
                segmented_face = Image.fromarray(segmented_np)
                face_tensor = preprocess(segmented_face).unsqueeze(0).to(device)
                with torch.no_grad():
                    embedding = model(face_tensor)
                embeddings.append(embedding.squeeze(0))
            if embeddings:
                avg_embedding = torch.stack(embeddings).mean(dim=0)
                known_faces[name] = avg_embedding
                print(f"[INFO] Saved images and embedding for {name}")
                torch.save(known_faces, known_faces_path)
                print(f"[INFO] Updated known faces at {known_faces_path}")
                self.name_label.config(text=f"Face for {name} saved and added to known faces.")
        else:
            print("[WARN] Name is empty. Skipping save.")
        self.prompted_for_name = False
        self.capturing_unknown = False
        self.capture_stage = 0
        self.captured_angles = []
        self.name_entry.destroy()
        self.submit_btn.destroy()
    def update_frame(self):
        if not self.running or not self.cap:
            return
        ret, frame = self.cap.read()
        if not ret:
            self.root.after(10, self.update_frame)
            return
        display_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(display_frame)
        imgtk = ImageTk.PhotoImage(image=pil_img)
        self.video_label.imgtk = imgtk
        self.video_label.configure(image=imgtk)
        current_time = time.time()
        if current_time - self.last_update_time > 2:
            self.last_update_time = current_time
            threading.Thread(target=self.process_frame, args=(frame.copy(),)).start()

        for (left, top, right, bottom), identity in self.last_predictions:
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, identity, (left, top - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        self.root.after(10, self.update_frame)
    def process_frame(self, frame):
        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            enhanced_frame = enhance_image(rgb_frame)
            enhanced_frame = upscale_if_needed(enhanced_frame)
            pil_frame = Image.fromarray(enhanced_frame)

            try:
                boxes, probs = detector.detect(pil_frame, landmarks=False)
                if boxes is None or len(boxes) == 0:
                    print("[DEBUG] No faces detected.")
                else:
                    print(f"[DEBUG] Detected {len(boxes)} face(s).")
            except Exception as e:
                print("[ERROR] Detector failed:", e)
                boxes, probs = None, None
            predictions = []
            if boxes is not None:
                for i, box in enumerate(boxes):
                    if probs[i] < 0.9:
                        continue
                    left, top, right, bottom = map(int, box)
                    face_crop = pil_frame.crop((left, top, right, bottom))
                    segmented_np = segment_face(face_crop)
                    segmented_face = Image.fromarray(segmented_np)

                    try:
                        face_tensor = preprocess(segmented_face).unsqueeze(0).to(device)
                        with torch.no_grad():
                            embedding = model(face_tensor)

                        identity = match_face(embedding[0], known_faces, threshold=0.65)
                        print(f"[DEBUG] Detected Identity: {identity}")
                    except Exception as e:
                        print("[ERROR] Face recognition failed:", e)
                        identity = "Unknown"

                    predictions.append(((left, top, right, bottom), identity))
                    self.name_label.config(text=f"Detected: {identity}")

                    if identity == "Unknown":
                        if not self.capturing_unknown:
                            self.capturing_unknown = True
                            self.capture_start_time = time.time()
                            self.capture_stage = 0
                            self.captured_angles = []
                            print("[INFO] Unknown detected. Starting capture process...")

                        if self.capture_stage < 3 and (time.time() - self.capture_start_time) > self.capture_stage * 2:
                            face_img = frame[top:bottom, left:right]
                            self.captured_angles.append(face_img)
                            print(f"[INFO] Captured angle {self.capture_stage + 1}")
                            self.capture_stage += 1

                        if self.capture_stage == 3 and not self.prompted_for_name:
                            self.prompt_for_name()

            self.last_predictions = predictions

        except Exception as e:
            print("[ERROR] process_frame failed:", e)


if __name__ == "__main__":
    root = tk.Tk()
    app = FaceRecognitionApp(root)
    root.protocol("WM_DELETE_WINDOW", app.stop_recognition)
    app.start_recognition()
    root.mainloop()
