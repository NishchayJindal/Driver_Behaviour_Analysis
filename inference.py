import streamlit as st
from ultralytics import YOLO
from pathlib import Path
import tempfile
import cv2
import numpy as np
from PIL import Image
from collections import Counter

st.set_page_config(page_title="ADAS Traffic Sign Detection", layout="wide")

# =========================================
# CONFIG
# =========================================
DEFAULT_MODEL_PATH = r"C:\nish\runs\detect\gtsdb_v8s\weights\best.pt"
CONF_DEFAULT = 0.35
IMG_SIZE_DEFAULT = 640


# Speed limits map (from GTSDB class names)
SPEED_LIMITS = {
    "speed_limit_20": 20,
    "speed_limit_30": 30,
    "speed_limit_50": 50,
    "speed_limit_60": 60,
    "speed_limit_70": 70,
    "speed_limit_80": 80,
    "speed_limit_100": 100,
    "speed_limit_120": 120,
}

# Other violation-type signs
VIOLATION_SIGNS = {
    "no_entry": "Driving into a NO ENTRY area!",
    "no_overtaking": "Overtaking in NO OVERTAKING zone!",
    "no_overtaking_trucks": "Truck overtaking violation!",
    "no_parking": "NO PARKING zone violation!",
    "no_stopping": "NO STOPPING zone violation!",
}


# =========================================
# HELPERS
# =========================================

@st.cache_resource
def load_model(path: str):
    """Cached YOLO model loader."""
    return YOLO(path)


def draw_detections(frame_bgr, result, class_names,
                    vehicle_speed_kmh,
                    total_sign_counts: Counter,
                    total_violation_counts: Counter):
    """
    Draw boxes + compute violations for a single frame.
    Returns annotated frame (BGR) and latest warning text (if any).
    """
    vis = frame_bgr.copy()
    latest_warning = None
    current_speed_limit = None

    if result.boxes is not None:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cid = int(box.cls[0])
            conf = float(box.conf[0])

            label = class_names.get(cid, str(cid))
            total_sign_counts[label] += 1

            # Draw bounding box
            color = (0, 255, 0)  # green
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)

            # Box label text
            text = f"{label} {conf:.2f}"
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(vis, (x1, y1 - th - 4), (x1 + tw + 6, y1), color, -1)
            cv2.putText(vis, text, (x1 + 3, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

            # --- Speed-limit logic ---
            if label in SPEED_LIMITS:
                limit = SPEED_LIMITS[label]
                if current_speed_limit is None or limit < current_speed_limit:
                    current_speed_limit = limit

            # --- Other violation signs ---
            if label in VIOLATION_SIGNS:
                msg = VIOLATION_SIGNS[label]
                total_violation_counts[msg] += 1
                latest_warning = msg

    # --- Frame-level warnings / info overlay ---
    y_text = 40
    if current_speed_limit is not None:
        if vehicle_speed_kmh > current_speed_limit:
            msg = f"SPEED LIMIT {current_speed_limit} VIOLATED! ({vehicle_speed_kmh} km/h)"
            total_violation_counts[msg] += 1
            latest_warning = msg
            cv2.putText(vis, msg, (30, y_text),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
            y_text += 30
        else:
            msg = f"Speed limit {current_speed_limit} km/h - OK"
            cv2.putText(vis, msg, (30, y_text),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            y_text += 30

    if latest_warning and current_speed_limit is None:
        # Draw non-speed warnings if not already drawn
        cv2.putText(vis, latest_warning, (30, y_text),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)

    return vis, latest_warning


def run_video_inference(model, video_path, conf, imgsz, vehicle_speed_kmh, progress_callback=None):
    """Run ADAS-style detection+violation checks over a video."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("Unable to open video.")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 1)

    tmp_out = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    output_path = tmp_out.name
    tmp_out.close()

    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    class_names = {int(k): str(v) for k, v in model.names.items()}
    sign_counts = Counter()
    violation_counts = Counter()

    frame_idx = 0
    preview_img = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1

        results = model.predict(source=frame, conf=conf, imgsz=imgsz, verbose=False)[0]

        vis_bgr, latest_warning = draw_detections(
            frame, results, class_names,
            vehicle_speed_kmh,
            sign_counts,
            violation_counts
        )

        writer.write(vis_bgr)

        # progress
        if progress_callback:
            progress_callback(frame_idx / total_frames)

        if frame_idx % 20 == 0:
            preview_img = Image.fromarray(cv2.cvtColor(vis_bgr, cv2.COLOR_BGR2RGB))

    cap.release()
    writer.release()

    return output_path, frame_idx, preview_img, sign_counts, violation_counts


# =========================================
# STREAMLIT UI
# =========================================

st.title("🚗 ADAS Traffic Sign Detection & Violation Warning")

col1, col2 = st.columns([1, 2])

with col1:
    model_path = st.text_input("YOLO model (.pt)", DEFAULT_MODEL_PATH)
    conf = st.slider("Confidence threshold", 0.1, 1.0, CONF_DEFAULT)
    imgsz = st.selectbox("Image size", [320, 416, 512, 640, 960], index=3)
    vehicle_speed = st.number_input("Vehicle speed (km/h)", min_value=0, max_value=200, value=70, step=5)
    upload = st.file_uploader("Upload Image or Video", type=["jpg", "jpeg", "png", "mp4", "avi", "mov", "mkv"])
    run_button = st.button("Run ADAS Detection")

with col2:
    st.info("Model is cached after first load for faster inference.\n"
            "Violations checked: Speed limits, no-entry, no-overtaking, no-parking, no-stopping.")


# --------- LOAD MODEL ----------
model = None
CLASS_NAMES = {}

if model_path:
    try:
        with st.spinner("Loading model..."):
            model = load_model(model_path)
            CLASS_NAMES = {int(k): str(v) for k, v in model.names.items()}
    except Exception as e:
        st.error(f"❌ Failed to load model: {e}")
        model = None


# --------- RUN DETECTION ----------
if run_button and upload and model:
    ext = upload.name.lower().split(".")[-1]
    tmp = tempfile.NamedTemporaryFile(suffix=f".{ext}", delete=False)
    tmp.write(upload.read())
    tmp.close()
    file_path = tmp.name

    st.subheader("Model Classes")
    st.json(CLASS_NAMES)

    # ----- IMAGE MODE -----
    if ext in ["jpg", "jpeg", "png"]:
        st.subheader("Image Result")

        img = Image.open(file_path).convert("RGB")
        bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

        results = model.predict(source=bgr, conf=conf, imgsz=imgsz, verbose=False)[0]

        sign_counts = Counter()
        violation_counts = Counter()

        vis_bgr, latest_warning = draw_detections(
            bgr, results, CLASS_NAMES,
            vehicle_speed,
            sign_counts,
            violation_counts
        )

        st.image(cv2.cvtColor(vis_bgr, cv2.COLOR_BGR2RGB), caption="Annotated Image")

        st.subheader("Detected Signs")
        st.json(dict(sign_counts))

        st.subheader("Violations")
        if violation_counts:
            st.error("Violations detected in this image:")
            st.json(dict(violation_counts))
        else:
            st.success("No violations detected for this image (with given speed).")

    # ----- VIDEO MODE -----
    else:
        st.subheader("Video Result")
        progress = st.progress(0.0)

        out_path, frames, preview, sign_counts, violation_counts = run_video_inference(
            model, file_path, conf, imgsz, vehicle_speed,
            progress_callback=lambda p: progress.progress(p)
        )

        st.success(f"Processed {frames} frames.")

        if preview:
            st.image(preview, caption="Sample annotated frame")

        st.subheader("Detected Signs in Video")
        st.json(dict(sign_counts))

        st.subheader("Violations in Video")
        if violation_counts:
            st.error("Violations detected in this video:")
            st.json(dict(violation_counts))
        else:
            st.success("No violations detected in this video (with given speed).")

        # show and download result video
        with open(out_path, "rb") as f:
            st.video(f.read())
        with open(out_path, "rb") as f:
            st.download_button("Download Annotated Video", f.read(), "adas_output.mp4")

elif run_button and not upload:
    st.warning("Please upload an image or video first.")
elif run_button and not model:
    st.warning("Model not loaded. Check your model path.")
