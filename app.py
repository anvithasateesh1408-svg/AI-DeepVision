import cv2
import torch
import numpy as np
import streamlit as st
import time

from csrnet_model import CSRNet

# ================= CONFIG =================
st.set_page_config(page_title="DeepVision Crowd Monitor", layout="wide")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "csrnet_epoch105.pth"
CROWD_THRESHOLD = 70

# ================= LOAD MODEL =================
@st.cache_resource
def load_model():
    model = CSRNet().to(DEVICE)

    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    fixed_state = {k.replace("core.", ""): v for k, v in checkpoint.items()}

    model.load_state_dict(fixed_state, strict=True)
    model.eval()

    st.success("✅ Model loaded successfully (architecture matched)")
    return model


model = load_model()

# ================= PREPROCESS =================
def preprocess(frame):
    img = cv2.resize(frame, (640, 360))
    img = img.astype(np.float32) / 255.0

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    img = (img - mean) / std
    img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
    return img.to(DEVICE)

# ================= PROCESS FRAME =================
def process_frame(frame):
    h, w, _ = frame.shape
    img = preprocess(frame)

    with torch.no_grad():
        density = model(img)

    density_map = density.squeeze().cpu().numpy()
    density_map[density_map < 0] = 0

    count = int(density_map.sum())

    density_map = cv2.resize(density_map, (w, h))
    density_norm = density_map / density_map.max() if density_map.max() > 0 else density_map

    heatmap = cv2.applyColorMap(
        (density_norm * 255).astype(np.uint8),
        cv2.COLORMAP_JET
    )

    overlay = cv2.addWeighted(frame, 0.75, heatmap, 0.25, 0)

    # Banner
    cv2.rectangle(overlay, (0, 0), (w, 50), (0, 0, 0), -1)
    cv2.putText(overlay, f"Count: {count}", (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    status = "CROWD ALERT" if count >= CROWD_THRESHOLD else "CROWD NORMAL"
    color  = (0, 0, 255) if count >= CROWD_THRESHOLD else (0, 255, 0)

    cv2.putText(overlay, status, (260, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)

    return overlay, count

# ================= UI =================
st.title("🧠 DeepVision Crowd Monitor – Milestone 3")

mode = st.radio("Select Input Mode", ["Webcam", "Upload Video"])

frame_box = st.image([])
count_box = st.empty()
alert_box = st.empty()

# ================= WEBCAM =================
if mode == "Webcam":
    start = st.checkbox("▶ Start Webcam")

    if start:
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

        while cap.isOpened() and start:
            ret, frame = cap.read()
            if not ret:
                break

            overlay, count = process_frame(frame)
            frame_box.image(overlay, channels="BGR")
            count_box.metric("👥 Crowd Count", count)

            if count >= CROWD_THRESHOLD:
                alert_box.error("🚨 OVERCROWDING DETECTED")
            else:
                alert_box.success("✅ Crowd Level Normal")

        cap.release()

# ================= VIDEO =================
if mode == "Upload Video":
    video = st.file_uploader("Upload a video", type=["mp4", "avi"])

    if video:
        with open("temp.mp4", "wb") as f:
            f.write(video.read())

        cap = cv2.VideoCapture("temp.mp4")
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        start_time = time.time()
        frame_id = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_id += 1
            overlay, count = process_frame(frame)

            elapsed = time.time() - start_time
            progress = (frame_id / total) * 100
            eta = (elapsed / frame_id) * (total - frame_id) if frame_id > 0 else 0

            frame_box.image(overlay, channels="BGR")
            count_box.markdown(
                f"""
🎬 **Frame:** {frame_id}/{total}  
📊 **Progress:** {progress:.1f}%  
👥 **Count:** {count}  
⏳ **Elapsed:** {elapsed:.1f}s  
⏱ **ETA:** {eta:.1f}s
"""
            )

            if count >= CROWD_THRESHOLD:
                alert_box.error("🚨 OVERCROWDING DETECTED")
            else:
                alert_box.success("✅ Crowd Level Normal")

        cap.release()
        st.success("🎉 Video processed successfully!")
