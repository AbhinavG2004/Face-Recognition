import os
import cv2
import time
import torch
import torch.nn as nn
import numpy as np
import pickle
from torchvision import transforms
from facenet_pytorch import MTCNN
from PIL import Image, ImageTk
import tkinter as tk

# ----------------------------------
# Set device to GPU if available
# ----------------------------------
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Using device:", device)

# ----------------------------------
# Image Enhancement Functions (with CLAHE)
# ----------------------------------
def enhance_image(img_rgb, alpha=1.3, beta=15):
    """
    Enhance an image by adjusting brightness/contrast and then applying CLAHE
    (Contrast Limited Adaptive Histogram Equalization) on the luminance channel.
    """
    # Convert RGB to BGR for OpenCV processing
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    # Global brightness/contrast adjustment
    adjusted = cv2.convertScaleAbs(img_bgr, alpha=alpha, beta=beta)
    # Convert to LAB color space for CLAHE on the L-channel
    lab = cv2.cvtColor(adjusted, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    merged = cv2.merge((cl, a, b))
    equalized = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
    # Convert back to RGB
    final_rgb = cv2.cvtColor(equalized, cv2.COLOR_BGR2RGB)
    return final_rgb

def upscale_if_needed(img, min_width=224, min_height=224):
    """
    Upscale the image if its dimensions are below the given minimum.
    """
    height, width = img.shape[:2]
    if width < min_width or height < min_height:
        scale = max(min_width/width, min_height/height)
        new_width = int(width * scale)
        new_height = int(height * scale)
        return cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
    return img

# ----------------------------------
# 1. Define the Face Feature Extractor Model using ResNet50
# ----------------------------------
class FaceClassifier(nn.Module):
    def __init__(self, dropout_rate=0.5, feature_extract=True):
        super(FaceClassifier, self).__init__()
        self.feature_extract = feature_extract
        # Load a pretrained ResNet50 model
        self.backbone = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)
        in_features = self.backbone.fc.in_features
        # Replace the final FC layer with dropout and a new linear layer producing a 512-dim embedding
        self.backbone.fc = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(in_features, 512)
        )
        if self.feature_extract:
            self.feature_extractor = self.backbone

    def forward(self, x):
        if self.feature_extract:
            features = self.feature_extractor(x)
            return features  # shape: [batch, 512]
        else:
            return self.backbone(x)

# Instantiate the model and move to device
model = FaceClassifier(dropout_rate=0.5, feature_extract=True)
model.to(device)
model.eval()

# ----------------------------------
# 2. Define Preprocessing Transform (ResNet50 expects 224x224 images)
# ----------------------------------
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ----------------------------------
# 3. Define Face Segmentation using GrabCut
# ----------------------------------
def segment_face(face_img):
    """
    Uses GrabCut to segment the face from the background.
    Expects face_img as a PIL Image.
    Returns a segmented image (numpy array in RGB).
    """
    img = np.array(face_img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    mask = np.zeros(img.shape[:2], np.uint8)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    height, width = img.shape[:2]
    rect = (1, 1, width - 2, height - 2)
    cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    segmented = img * mask2[:, :, np.newaxis]
    segmented = cv2.cvtColor(segmented, cv2.COLOR_BGR2RGB)
    return segmented

# ----------------------------------
# 4. Build or Load Known Faces Database
# ----------------------------------
DATABASE_PATH = r"C:\Users\Monish V\OneDrive\Documents\RANDOM_PROJECTS\Face_detction\Dataset\Original Images\Original Images"
KNOWN_FACES_FILE = "known_faces.pkl"
MODEL_FILE = "face_feature_extractor.pth"

from facenet_pytorch import MTCNN
detector = MTCNN(keep_all=True, device=device)

def build_known_faces(database_path, model, preprocess, detector, device='cuda'):
    known_faces = {}  # identity -> embedding vector (numpy array)
    for identity in os.listdir(database_path):
        identity_path = os.path.join(database_path, identity)
        if not os.path.isdir(identity_path):
            continue
        embeddings = []
        print(f"Processing identity: {identity}")
        for filename in os.listdir(identity_path):
            file_path = os.path.join(identity_path, filename)
            try:
                img = Image.open(file_path).convert('RGB')
            except Exception as e:
                print(f"    Error opening image {filename}: {e}")
                continue
            boxes, probs = detector.detect(img, landmarks=False)
            if boxes is None or len(boxes) == 0:
                continue
            idx = np.argmax(probs)
            box = boxes[idx]
            left, top, right, bottom = map(int, box)
            face_crop = img.crop((left, top, right, bottom))
            segmented_face_np = segment_face(face_crop)
            segmented_face = Image.fromarray(segmented_face_np)
            try:
                face_tensor = preprocess(segmented_face).unsqueeze(0).to(device)
            except Exception as e:
                print(f"    Error processing tensor for {filename}: {e}")
                continue
            with torch.no_grad():
                emb = model(face_tensor)
            embeddings.append(emb.cpu().numpy()[0])
        if embeddings:
            avg_emb = np.mean(embeddings, axis=0)
            avg_emb /= np.linalg.norm(avg_emb)
            known_faces[identity] = avg_emb
            print(f"Processed {identity}: {len(embeddings)} faces")
    return known_faces

if os.path.exists(KNOWN_FACES_FILE):
    with open(KNOWN_FACES_FILE, "rb") as f:
        known_faces = pickle.load(f)
    print("Loaded known faces database from file.")
else:
    print("Building known faces database...")
    known_faces = build_known_faces(DATABASE_PATH, model, preprocess, detector, device)
    with open(KNOWN_FACES_FILE, "wb") as f:
        pickle.dump(known_faces, f)
    print("Saved known faces database to file.")

if not os.path.exists(MODEL_FILE):
    torch.save(model.state_dict(), MODEL_FILE)
    print("Saved face feature extractor model.")
else:
    print("Model state already saved.")

# ----------------------------------
# 5. Face Matching Function
# ----------------------------------
def match_face(face_embedding, known_db, threshold=0.8):
    face_emb = face_embedding.cpu().detach().numpy()
    face_emb /= np.linalg.norm(face_emb)
    best_match = "Unknown"
    best_sim = -1
    for name, known_emb in known_db.items():
        sim = np.dot(face_emb, known_emb)  # cosine similarity
        if sim > best_sim:
            best_sim = sim
            best_match = name
    if best_sim < threshold:
        return "Unknown"
    return best_match

# ----------------------------------
# Warm-up and Delay for Consistent Inference
# ----------------------------------
def warmup_model(model, preprocess, device, num_iterations=5):
    dummy_input = torch.randn((1, 3, 224, 224)).to(device)
    print("Warming up model...")
    for _ in range(num_iterations):
        with torch.no_grad():
            _ = model(dummy_input)
    print("Warm-up complete.")

warmup_model(model, preprocess, device)
print("Waiting for 15 seconds before starting live recognition...")
time.sleep(15)

# ----------------------------------
# 6. GUI using Tkinter for Live Face Recognition with Prediction Smoothing
# ----------------------------------
class FaceRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Live Face Recognition with Smoothing")
        self.video_label = tk.Label(root)
        self.video_label.pack()
        self.btn_frame = tk.Frame(root)
        self.btn_frame.pack(fill="x")
        self.start_btn = tk.Button(self.btn_frame, text="Start Recognition", command=self.start_recognition)
        self.start_btn.pack(side="left", padx=10, pady=10)
        self.stop_btn = tk.Button(self.btn_frame, text="Stop", command=self.stop_recognition, state="disabled")
        self.stop_btn.pack(side="left", padx=10, pady=10)
        self.cap = None
        self.running = False
        self.last_update_time = 0
        self.last_predictions = []  # List of tuples: (box, identity)

    def start_recognition(self):
        if self.cap is None:
            self.cap = cv2.VideoCapture(0)
        self.running = True
        self.start_btn.config(state="disabled")
        self.stop_btn.config(state="normal")
        self.last_update_time = time.time()
        self.last_predictions = []
        self.update_frame()

    def stop_recognition(self):
        self.running = False
        self.start_btn.config(state="normal")
        self.stop_btn.config(state="disabled")
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def update_frame(self):
        if self.cap is None or not self.running:
            return
        ret, frame = self.cap.read()
        if not ret:
            self.root.after(10, self.update_frame)
            return

        # Convert frame from BGR to RGB, enhance, and upscale
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        enhanced_frame = enhance_image(rgb_frame)
        enhanced_frame = upscale_if_needed(enhanced_frame)
        pil_frame = Image.fromarray(enhanced_frame)

        # Update predictions every 2 seconds to smooth output
        current_time = time.time()
        if current_time - self.last_update_time > 2:
            boxes, probs, landmarks = detector.detect(pil_frame, landmarks=True)
            predictions = []
            if boxes is not None:
                for i, box in enumerate(boxes):
                    if probs[i] < 0.90:
                        continue
                    left, top, right, bottom = map(int, box)
                    face_crop = pil_frame.crop((left, top, right, bottom))
                    segmented_np = segment_face(face_crop)
                    segmented_face = Image.fromarray(segmented_np)
                    try:
                        face_tensor = preprocess(segmented_face).unsqueeze(0).to(device)
                    except Exception as e:
                        continue
                    with torch.no_grad():
                        embedding = model(face_tensor)
                    identity = match_face(embedding[0], known_faces, threshold=0.8)
                    predictions.append(((left, top, right, bottom), identity))
            self.last_predictions = predictions
            self.last_update_time = current_time

        # Draw the stored (smoothed) predictions on the current frame
        for (left, top, right, bottom), identity in self.last_predictions:
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, identity, (left, top - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(frame_rgb)
        imgtk = ImageTk.PhotoImage(image=pil_img)
        self.video_label.imgtk = imgtk
        self.video_label.configure(image=imgtk)
        self.root.after(10, self.update_frame)

# ----------------------------------
# Run the GUI Application
# ----------------------------------
if __name__ == "__main__":
    root = tk.Tk()
    app = FaceRecognitionApp(root)
    root.protocol("WM_DELETE_WINDOW", app.stop_recognition)
    root.mainloop()
