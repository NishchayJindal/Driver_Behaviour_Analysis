# adas_targeted_fixed.py
import streamlit as st
from ultralytics import YOLO
import cv2, tempfile, shutil, time, os
import numpy as np
from PIL import Image
from collections import Counter, deque
from pathlib import Path

st.set_page_config(page_title="ADAS - targeted violations", layout="wide")

DEFAULT_MODEL = r"C:\nish\runs\detect\gtsdb_v8s\weights\best.pt"
CONF_DEFAULT = 0.35

# class ids (from your yaml)
CLASS_IDS = {
    "stop": 13,
    "no_parking": 41,
    "turn_right": 30,
    "turn_left": 31,
    # speed limits are multiple labels (names mapped in model.names)
}

# speed mapping by name - will use model.names to detect which speed sign it is
SPEED_LIMITS_BY_NAME = {
    "speed_limit_20": 20,
    "speed_limit_30": 30,
    "speed_limit_50": 50,
    "speed_limit_60": 60,
    "speed_limit_70": 70,
    "speed_limit_80": 80,
    "speed_limit_100": 100,
    "speed_limit_120": 120,
}

@st.cache_resource
def load_model(path):
    return YOLO(path)

# optical flow helper (sparse LK)
def estimate_motion(prev_gray, gray):
    """
    Always returns a 3-tuple: (motion_median_disp, horiz_median, gray)
    If prev_gray is None or tracking fails, returns (0.0, 0.0, gray).
    """
    if prev_gray is None:
        return 0.0, 0.0, gray
    p0 = cv2.goodFeaturesToTrack(prev_gray, mask=None, maxCorners=80, qualityLevel=0.3, minDistance=7, blockSize=7)
    if p0 is None:
        return 0.0, 0.0, gray
    p1, stt, err = cv2.calcOpticalFlowPyrLK(prev_gray, gray, p0, None)
    if p1 is None or stt is None:
        return 0.0, 0.0, gray
    good_new = p1[stt.flatten()==1]
    good_old = p0[stt.flatten()==1]
    if len(good_new) == 0:
        return 0.0, 0.0, gray
    disp = np.linalg.norm(good_new - good_old, axis=1)
    horiz = np.median((good_new - good_old)[:, 0])
    return float(np.median(disp)), float(horiz), gray

# draw helper
def draw_boxes(frame, results, names):
    vis = frame.copy()
    detected = []
    if getattr(results, "boxes", None) is not None:
        for box in results.boxes:
            x1,y1,x2,y2 = map(int, box.xyxy[0])
            cid = int(box.cls[0])
            conf = float(box.conf[0])
            nm = names.get(cid, str(cid))
            detected.append((nm, cid, (x1,y1,x2,y2), conf))
            cv2.rectangle(vis, (x1,y1),(x2,y2),(0,255,0),2)
            txt = f"{nm} {conf:.2f}"
            (tw,th),_ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.6,1)
            cv2.rectangle(vis,(x1,y1-th-4),(x1+tw+6,y1),(0,255,0),-1)
            cv2.putText(vis, txt, (x1+3,y1-5), cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,0,0),2)
    return vis, detected

# violation checks (targeted)
def check_targeted_violations_frame(detected, vehicle_speed_kmh, frame_w):
    messages = []
    violated = []
    # speed limits: pick smallest limit seen (closest applicable)
    current_limit = None
    for nm,cid,box,conf in detected:
        if nm in SPEED_LIMITS_BY_NAME:
            l = SPEED_LIMITS_BY_NAME[nm]
            if current_limit is None or l < current_limit:
                current_limit = l
    if current_limit is not None:
        if vehicle_speed_kmh > current_limit:
            msg = f"SPEED LIMIT {current_limit} violated ({vehicle_speed_kmh:.1f} km/h)"
            messages.append(("violation", msg)); violated.append(msg)
        else:
            messages.append(("ok", f"Speed OK (limit {current_limit})"))

    # stop & no_parking immediate heuristics for image mode
    for nm,cid,box,conf in detected:
        if cid == CLASS_IDS["stop"]:
            # image mode: if vehicle speed > 5 => mark violation
            if vehicle_speed_kmh > 5:
                msg = "STOP sign violated (vehicle speed > 5 km/h)"
                messages.append(("violation", msg)); violated.append(msg)
            else:
                messages.append(("ok", "STOP sign respected (low speed)"))
        if cid == CLASS_IDS["no_parking"]:
            # image mode: if speed < 5 treat as parked => violation
            if vehicle_speed_kmh < 5:
                msg = "NO PARKING violated (vehicle appears stopped/parked)"
                messages.append(("violation", msg)); violated.append(msg)
            else:
                messages.append(("ok", "No parking sign present"))
        if cid in (CLASS_IDS["turn_left"], CLASS_IDS["turn_right"]):
            messages.append(("info", f"Turn sign detected: {nm}"))
    return messages, violated

# Advanced per-video logic (temporal + optical flow)
class VideoState:
    def __init__(self,fps):
        self.prev_gray = None
        self.fps = fps if fps>0 else 25
        self.frame_idx = 0
        self.turn_history = deque(maxlen=30)   # store horiz motion over recent frames
        self.stop_event = None
        self.no_parking_since = None
        self.no_parking_zone = False

    def update(self, gray, t_sec):
        # estimate_motion now returns (motion, horiz, gray) consistently
        motion, horiz, new_prev = estimate_motion(self.prev_gray, gray)
        self.prev_gray = new_prev
        self.frame_idx += 1
        self.turn_history.append(horiz)
        return motion, horiz

def advanced_video_violation_logic(state, detections, motion, horiz, vehicle_speed_kmh, frame_w, t_sec, violation_counts):
    msgs = []
    # TURN sign logic:
    if any(d[1]==CLASS_IDS["turn_left"] for d in detections):
        med = np.median(state.turn_history) if state.turn_history else 0.0
        if med > -0.6:
            m = "TURN LEFT sign violated (no left-turn motion detected)"
            msgs.append(("violation", m)); violation_counts[m]+=1
        else:
            msgs.append(("ok","Turn-left respected"))
    if any(d[1]==CLASS_IDS["turn_right"] for d in detections):
        med = np.median(state.turn_history) if state.turn_history else 0.0
        if med < 0.6:
            m = "TURN RIGHT sign violated (no right-turn motion detected)"
            msgs.append(("violation", m)); violation_counts[m]+=1
        else:
            msgs.append(("ok","Turn-right respected"))

    # STOP sign: start/resolve/violation heuristics
    if any(d[1]==CLASS_IDS["stop"] for d in detections):
        if state.stop_event is None:
            state.stop_event = {"start": t_sec, "low_since": None, "resolved": False}
    if state.stop_event is not None:
        ev = state.stop_event
        if motion < 0.4:
            if ev["low_since"] is None:
                ev["low_since"] = t_sec
        else:
            ev["low_since"] = None
        if ev["low_since"] is not None and (t_sec - ev["low_since"]) > 0.5:
            ev["resolved"] = True
            msgs.append(("ok","STOP respected"))
        if (t_sec - ev["start"]) > 3.0 and not ev["resolved"]:
            m = "STOP sign violated (no sufficient stop after STOP)"
            msgs.append(("violation", m)); violation_counts[m]+=1
            ev["resolved"] = True
        if (t_sec - ev["start"]) > 4.5:
            state.stop_event = None

    # NO PARKING advanced
    if any(d[1]==CLASS_IDS["no_parking"] for d in detections):
        state.no_parking_zone = True
    if state.no_parking_zone:
        if motion < 0.4:
            if state.no_parking_since is None:
                state.no_parking_since = t_sec
        else:
            state.no_parking_since = None
        if state.no_parking_since is not None and (t_sec - state.no_parking_since) >= 3.0:
            m = "NO PARKING violated (vehicle stopped > 3s in no-parking zone)"
            msgs.append(("violation", m)); violation_counts[m]+=1
            state.no_parking_since = None
    return msgs

# UI
col1,col2 = st.columns([1,2])
with col1:
    model_path = st.text_input("Model (.pt)", DEFAULT_MODEL)
    conf = st.slider("Confidence", 0.1, 1.0, CONF_DEFAULT)
    imgsz = st.selectbox("Image size", [320,416,512,640,960], index=3)
    vehicle_speed = st.number_input("Vehicle speed (km/h)", min_value=0, max_value=200, value=50)
    upload = st.file_uploader("Upload Image or Video", type=["jpg","jpeg","png","mp4","avi","mov","mkv"])
    run = st.button("Run")
with col2:
    st.info("Only speed limits, stop, no_parking, turn_left, turn_right will produce violations.")

model = None
NAMES = {}
if model_path:
    try:
        with st.spinner("Loading model..."):
            model = load_model(model_path)
            NAMES = {int(k):str(v) for k,v in model.names.items()}
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        model = None

if run and upload and model:
    ext = upload.name.lower().split(".")[-1]
    tmp = tempfile.NamedTemporaryFile(suffix=f".{ext}", delete=False)
    tmp.write(upload.read()); tmp.close()
    fpath = tmp.name

    st.subheader("Model classes")
    st.json(NAMES)

    if ext in ("jpg","jpeg","png"):
        img = Image.open(fpath).convert("RGB")
        bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        res = model.predict(source=bgr, conf=conf, imgsz=imgsz, verbose=False)[0]
        vis, detections = draw_boxes(bgr, res, NAMES)
        msgs, violated = check_targeted_violations_frame(detections, vehicle_speed, bgr.shape[1])
        st.image(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
        for lvl,m in msgs:
            if lvl=="violation":
                st.error(m)
            elif lvl=="ok":
                st.success(m)
            else:
                st.info(m)

    else:
        st.subheader("Video (advanced)")
        prog = st.progress(0.0)
        cap = cv2.VideoCapture(fpath)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 1)

        tmpf = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        outp = tmpf.name
        tmpf.close()

        # try mp4v; fallback could be attempted by user if this fails
        writer = cv2.VideoWriter(outp, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w,h))
        if not writer.isOpened():
            st.error("VideoWriter failed to open — try changing FOURCC (e.g., 'XVID') or check codec support.")
            cap.release()
            writer.release()
        else:
            state = VideoState(fps)
            violation_counts = Counter()
            frame_idx = 0
            written_frames = 0
            preview = None

            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame_idx += 1
                t_sec = frame_idx / fps
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                # update state (calls estimate_motion internally)
                motion, horiz = state.update(gray, t_sec)

                # model prediction & draw
                res = model.predict(source=frame, conf=conf, imgsz=imgsz, verbose=False)[0]
                vis, detections = draw_boxes(frame, res, NAMES)

                # prepare detection tuples (nm,cid,box,conf)
                dets = []
                if getattr(res, "boxes", None) is not None:
                    for box in res.boxes:
                        x1,y1,x2,y2 = map(int, box.xyxy[0])
                        cid = int(box.cls[0])
                        nm = NAMES.get(cid, str(cid))
                        confv = float(box.conf[0])
                        dets.append((nm, cid, (x1,y1,x2,y2), confv))

                # advanced logic
                adv_msgs = advanced_video_violation_logic(state, dets, motion, horiz, vehicle_speed, w, t_sec, violation_counts)
                # speed-limit check in video mode
                for nm,cid,box,cf in dets:
                    if nm in SPEED_LIMITS_BY_NAME:
                        limit = SPEED_LIMITS_BY_NAME[nm]
                        if vehicle_speed > limit:
                            m = f"SPEED LIMIT {limit} violated ({vehicle_speed} km/h)"
                            adv_msgs.append(("violation", m)); violation_counts[m]+=1

                # draw adv_msgs text
                y = 30
                for lvl, mm in adv_msgs:
                    col = (0,0,255) if lvl=="violation" else (0,255,255)
                    cv2.putText(vis, mm, (20,y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2)
                    y += 28

                writer.write(vis)
                written_frames += 1
                if frame_idx % 20 == 0:
                    preview = Image.fromarray(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
                # progress update (cap total may be 0 or wrong; guard division)
                prog.progress(min(frame_idx/ max(total,1), 1.0))

            cap.release()
            writer.release()

            # read output once and reuse
            if os.path.exists(outp):
                size = os.path.getsize(outp)
            else:
                size = 0

            data = None
            if size > 0:
                with open(outp, "rb") as f:
                    data = f.read()

            st.success(f"Processed {frame_idx} frames; wrote {written_frames} frames; output size={size} bytes")

            if preview:
                st.image(preview)

            st.subheader("Violations summary")
            if violation_counts:
                st.json(dict(violation_counts))
            else:
                st.info("No violations detected")

            if not data:
                st.error(f"Output file is empty (size={size} bytes). Check codec, writer release, or try different FOURCC.")
            else:
                # show video and provide download (use correct mime for mp4)
                st.video(data)
                st.download_button("Download annotated video", data, "annotated.mp4", mime="video/mp4")

elif run and not upload:
    st.warning("Upload an image or video.")
elif run and not model:
    st.warning("Load your .pt model first.")
