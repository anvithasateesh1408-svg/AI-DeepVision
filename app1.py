import os
import cv2
import torch
import gdown
import streamlit as st
import numpy as np

from csrnet_model import CSRNet
from utils_email import init_db, get_all_emails, send_alert_emails


# =========================================================
# CONFIG
# =========================================================
st.set_page_config(
    page_title="Crowd Monitoring System (CSRNet)",
    layout="wide"
)

DEVICE = "cpu"
DENSITY_ALERT_THRESHOLD = 70.0

MODEL_PATH = "csrnet_video_finetuned_final.pth"
GDRIVE_FILE_ID = "1ax1G5Q1s5lmD6MVa8w2EOU26gX4QCfaC"


# =========================================================
# DOWNLOAD MODEL (Google Drive)
# =========================================================
def download_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("⬇️ Downloading CSRNet model (one-time)..."):
            gdown.download(
                id=GDRIVE_FILE_ID,
                output=MODEL_PATH,
                quiet=False,
                fuzzy=True
            )

download_model()


# =========================================================
# INIT DATABASE
# =========================================================
init_db()


# =========================================================
# LOAD MODEL (SAFE FOR DEPLOYMENT)
# =========================================================
@st.cache_resource
def load_model():
    model = CSRNet()

    checkpoint = torch.load(MODEL_PATH, map_location="cpu")

    # Handle checkpoint / DataParallel cases
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    clean_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            clean_state_dict[k.replace("module.", "")] = v
        else:
            clean_state_dict[k] = v

    model.load_state_dict(clean_state_dict, strict=False)
    model.eval()
    return model


model = load_model()


# =========================================================
# PROCESS FRAME
# =========================================================
def process_frame(frame):
    H, W, _ = frame.shape

    img = cv2.resize(frame, (640, 360))
    img = img.astype(np.float32) / 255.0

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img = (img - mean) / std

    img_t = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)

    with torch.no_grad():
        density = torch.relu(model(img_t))

    density_map = density.squeeze().cpu().numpy()
    crowd_count = float(density_map.sum())

    # Visualization
    density_map = cv2.GaussianBlur(density_map, (13, 13), 0)
    density_map = cv2.resize(density_map, (W, H))

    p98 = np.percentile(density_map, 98)
    density_vis = np.clip(density_map / (p98 + 1e-6), 0, 1)

    heatmap = cv2.applyColorMap(
        (density_vis * 255).astype(np.uint8),
        cv2.COLORMAP_JET
    )

    overlay = cv2.addWeighted(frame, 0.7, heatmap, 0.3, 0)

    return overlay, crowd_count


# =========================================================
# VIDEO ESTIMATION
# =========================================================
def estimate_from_video(video_path, n_frames=10):
    cap = cv2.VideoCapture(video_path)

    counts = []
    last_overlay = None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(total_frames // n_frames, 1)

    idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if idx % step == 0:
            overlay, count = process_frame(frame)
            counts.append(count)
            last_overlay = overlay

        idx += 1
        if len(counts) >= n_frames:
            break

    cap.release()

    if not counts:
        return None, 0.0

    return last_overlay, float(np.mean(counts))


# =========================================================
# UI
# =========================================================
st.title("🧠 Crowd Monitoring System (CSRNet)")

uploaded_file = st.file_uploader(
    "Upload a crowd video",
    type=["mp4", "avi", "mov"]
)

if uploaded_file:
    with open("temp.mp4", "wb") as f:
        f.write(uploaded_file.read())

    overlay, crowd_count = estimate_from_video("temp.mp4")

    if overlay is not None:
        col1, col2 = st.columns([3, 1])

        with col1:
            st.image(overlay, caption="Crowd Density Map", width=700)

        with col2:
            st.metric("📊 Crowd Count", f"{crowd_count:.2f}")

            if crowd_count >= DENSITY_ALERT_THRESHOLD:
                st.error("⚠️ CROWD ALERT")
                msg = send_alert_emails(get_all_emails(), crowd_count)
                st.warning(msg)
            else:
                st.success("✅ SAFE")
