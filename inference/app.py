import streamlit as st
import cv2
import os
import time
import requests
import threading
from ultralytics import YOLO
import numpy as np
from collections import defaultdict
from datetime import datetime, timedelta
from PIL import Image
import torch
import torchvision

from datetime import datetime

# Get current date
current_date = datetime.now().strftime("%d-%m-%Y")

# ==== TELEGRAM CONFIG ====
BOT_TOKEN = 'YOUR TOKEN'
CHAT_ID = 'YOUR CHAT ID'

def send_telegram_alert(image_path, message):
    url = f'https://api.telegram.org/bot{BOT_TOKEN}/sendPhoto'
    with open(image_path, 'rb') as photo:
        files = {'photo': photo}
        data = {'chat_id': CHAT_ID, 'caption': message}
        response = requests.post(url, files=files, data=data)
        return response.status_code == 200

# ==== UTILS ====
def non_max_suppression_classwise(boxes, confs, iou_threshold=0.5):
    if len(boxes) == 0:
        return []
    boxes_tensor = torch.tensor(boxes, dtype=torch.float32)
    confs_tensor = torch.tensor(confs)
    keep = torchvision.ops.nms(boxes_tensor, confs_tensor, iou_threshold)
    return keep.numpy().tolist()

def save_violation_clip(image_path, clip_path, frame, frame_buffer, fps, width, height, message):
    cv2.imwrite(image_path, frame)
    writer = cv2.VideoWriter(clip_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    for f in frame_buffer:
        writer.write(f)
    writer.release()
    send_telegram_alert(image_path, message)

# ==== PATH SETUP ====
base_dir = '/Project/inference'
model_path = os.path.join(base_dir, 'models/MODEL.pt')
video_dir = os.path.join(base_dir, 'input_videos')
output_dir = os.path.join(base_dir, f'output_videos/output_videos_{current_date}')
incident_base = os.path.join(base_dir, 'incidents')
no_helmet_dir = os.path.join(incident_base, f'no_helmet_{current_date}')
triple_riding_dir = os.path.join(incident_base, f'triple_riding_{current_date}')
no_parking_dir = os.path.join(incident_base, f'no_parking_{current_date}')

for d in [no_helmet_dir, triple_riding_dir, no_parking_dir, output_dir]:
    os.makedirs(os.path.join(d, 'frames'), exist_ok=True)
    os.makedirs(os.path.join(d, 'clips'), exist_ok=True)

# ==== MODEL & LABELS ====
model = YOLO(model_path)
labels = ['person', 'car', 'bike', 'auto', 'bus', 'livestock', 'helmet', 'no_helmet', 'truck', 'number_plate']
colors = {
    'person': (0, 255, 0), 'car': (255, 0, 0), 'bike': (0, 0, 255),
    'auto': (255, 255, 0), 'bus': (255, 165, 0), 'livestock': (128, 0, 128),
    'helmet': (0, 255, 255), 'no_helmet': (0, 100, 255), 'truck': (139, 69, 19), 'number_plate': (255, 20, 147)
}

# ==== UI ====
st.sidebar.title("🧭 Navigation")
selected_tab = st.sidebar.radio("Go to", ["Live Inference", "Incidents", "Map"])

st.sidebar.markdown("### 🎯 Confidence Thresholds")
confidence_thresholds = {label: st.sidebar.slider(label, 0.0, 1.0, 0.3, 0.05) for label in labels}

# ==== POLYGONAL ROI (No Parking Zone) ====
NO_PARKING_POLYGONS = [
    np.array([[501, 1521], [744, 957], [1026, 960], [1149, 1512]], np.int32),
    np.array([[1494, 1029], [2748, 1305], [2973, 1047], [1695, 927]], np.int32)
]

# ==== Live Inference ====
if selected_tab == "Live Inference":
    st.title("🚦AI-Based Real Time Traffic Violation Detection System")
    input_mode = st.radio("Select Input", ["Upload Video", "Live Cam"])
    label_filter = st.multiselect("Filter Labels", labels, default=labels)
    
    violation_options = ["No Helmet", "Triple Riding", "No Parking"]
    selected_violations = st.multiselect("Select Violations to Monitor", violation_options, default=violation_options)

    if 'stop_camera' not in st.session_state:
        st.session_state['stop_camera'] = False

    cap = None
    if input_mode == "Upload Video":
        video_files = [f for f in os.listdir(video_dir) if f.endswith((".mp4", ".avi", ".MOV"))]
        selected_video = st.selectbox("Select Video", video_files)
        if st.button("Start Inference"):
            cap = cv2.VideoCapture(os.path.join(video_dir, selected_video))
    else:
        ip_url = st.text_input("Enter IP Camera RTSP URL", "RTSP LINK")
        if st.button("🎥 Start Camera"):
            cap = cv2.VideoCapture(ip_url)
            st.session_state['stop_camera'] = False
        if st.button("🛑 Stop Camera"):
            st.session_state['stop_camera'] = True

    if cap and cap.isOpened() and not st.session_state['stop_camera']:
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        output_path = os.path.join(output_dir, f'output_{datetime.now().strftime("%Y%m%d_%H%M%S")}.mp4')
        out_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        stframe = st.empty()
        frame_buffer = []
        summary = defaultdict(int)
        cooldown = 10
        last_detection_time = defaultdict(lambda: datetime.min)
        no_parking_trackers = defaultdict(lambda: {"count": 0, "last_seen": 0})
        current_frame_idx = 0
        stationary_time_threshold_seconds = 5

        while cap.isOpened() and not st.session_state['stop_camera']:
            ret, frame = cap.read()
            if not ret:
                break
            current_frame_idx += 1
            raw_frame = frame.copy()
            frame_buffer.append(raw_frame)
            if len(frame_buffer) > fps * 10:
                frame_buffer.pop(0)

            results = model(frame)[0]
            detections_by_label = defaultdict(list)
            boxes_by_label = defaultdict(list)
            for box in results.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = box.conf[0].item()
                cls = int(box.cls[0].item())
                label = labels[cls]
                if conf < confidence_thresholds[label] or label not in label_filter:
                    continue
                detections_by_label[label].append(((x1, y1, x2, y2), conf))

            for label, detections in detections_by_label.items():
                boxes, confs = zip(*detections)
                keep = non_max_suppression_classwise(boxes, confs)
                for idx in keep:
                    (x1, y1, x2, y2), conf = detections[idx]
                    summary[label] += 1
                    color = colors.get(label, (255, 255, 255))
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, f'{label} {conf:.2f}', (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    boxes_by_label[label].append((x1, y1, x2, y2))

            for roi in NO_PARKING_POLYGONS:
                cv2.polylines(frame, [roi], isClosed=True, color=(0, 0, 255), thickness=2)

            if "Triple Riding" in selected_violations:
                for bike in boxes_by_label['bike']:
                    persons = [p for p in boxes_by_label['person'] if bike[0] < (p[0]+p[2])//2 < bike[2] and bike[1] < (p[1]+p[3])//2 < bike[3]]
                    if len(persons) >= 3 and (datetime.now() - last_detection_time['triple_riding']).total_seconds() > cooldown:
                        last_detection_time['triple_riding'] = datetime.now()
                        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
                        img_path = os.path.join(triple_riding_dir, 'frames', f'triple_frame_{ts}.jpg')
                        clip_path = os.path.join(triple_riding_dir, 'clips', f'triple_incident_{ts}.mp4')
                        threading.Thread(target=save_violation_clip, args=(img_path, clip_path, frame.copy(), frame_buffer.copy(), fps, width, height, "🚨 Triple Riding Detected!")).start()

            if "No Helmet" in selected_violations:
                if 'no_helmet' in boxes_by_label and 'person' in boxes_by_label and 'bike' in boxes_by_label:
                    for nh_box in boxes_by_label['no_helmet']:
                        x_nh, y_nh = (nh_box[0] + nh_box[2]) // 2, (nh_box[1] + nh_box[3]) // 2
                        for person_box in boxes_by_label['person']:
                            if person_box[0] <= x_nh <= person_box[2] and person_box[1] <= y_nh <= person_box[3]:
                                x_p, y_p = (person_box[0] + person_box[2]) // 2, (person_box[1] + person_box[3]) // 2
                                for bike_box in boxes_by_label['bike']:
                                    if bike_box[0] <= x_p <= bike_box[2] and bike_box[1] <= y_p <= bike_box[3]:
                                        if (datetime.now() - last_detection_time['no_helmet']).total_seconds() > cooldown:
                                            last_detection_time['no_helmet'] = datetime.now()
                                            ts = datetime.now().strftime('%Y%m%d_%H%M%S')
                                            img_path = os.path.join(no_helmet_dir, 'frames', f'helmet_frame_{ts}.jpg')
                                            clip_path = os.path.join(no_helmet_dir, 'clips', f'helmet_incident_{ts}.mp4')
                                            threading.Thread(
                                                target=save_violation_clip,
                                                args=(img_path, clip_path, frame.copy(), frame_buffer.copy(), fps, width, height,
                                                    "🚨 No Helmet Violation Detected!")
                                            ).start()
                                            break

            if "No Parking" in selected_violations:
                for vlabel in ['car', 'bike', 'auto']:
                    for (x1, y1, x2, y2) in boxes_by_label[vlabel]:
                        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                        key = f"{vlabel}_{cx}_{cy}"
                        inside_any_roi = any(cv2.pointPolygonTest(roi, (cx, cy), False) >= 0 for roi in NO_PARKING_POLYGONS)
                        if inside_any_roi:
                            no_parking_trackers[key]['count'] += 1
                            no_parking_trackers[key]['last_seen'] = current_frame_idx
                        else:
                            no_parking_trackers[key]['count'] = 0

                for key, val in list(no_parking_trackers.items()):
                    if val['count'] > fps * stationary_time_threshold_seconds and (datetime.now() - last_detection_time['no_parking']).total_seconds() > cooldown:
                        last_detection_time['no_parking'] = datetime.now()
                        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
                        img_path = os.path.join(no_parking_dir, 'frames', f'park_frame_{ts}.jpg')
                        clip_path = os.path.join(no_parking_dir, 'clips', f'park_incident_{ts}.mp4')
                        threading.Thread(target=save_violation_clip, args=(img_path, clip_path, frame.copy(), frame_buffer.copy(), fps, width, height, "❌ No Parking Violation Detected!")).start()
                        del no_parking_trackers[key]

            out_writer.write(frame)
            stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels='RGB', use_container_width=True)
            time.sleep(1 / fps)

        cap.release()
        out_writer.release()
        st.success(f"✅ Output saved to {output_path}")
        st.write(dict(summary))





# ==== Incident View ====

import os
import streamlit as st
import math
from datetime import datetime
import plotly.express as px
from collections import Counter

# Mapping of violation types to folder prefixes
VIOLATION_TYPES = {
    "No Helmet": "no_helmet_",
    "Triple Riding": "triple_riding_",
    "No Parking": "no_parking_"
}


def load_incident_data():
    """
    Scan incident folder and organize image paths by (violation type, date).
    Returns:
        incident_map[(violation_type, date)] = list of (img_path, caption)
    """
    incident_map = {}

    for folder_name in os.listdir(incident_base):
        for violation_label, prefix in VIOLATION_TYPES.items():
            if folder_name.startswith(prefix):
                date_str = folder_name.replace(prefix, "")
                try:
                    date_obj = datetime.strptime(date_str, "%d-%m-%Y").date()
                    frame_dir = os.path.join(incident_base, folder_name, "frames")
                    if os.path.isdir(frame_dir):
                        for img_file in os.listdir(frame_dir):
                            if img_file.endswith(".jpg"):
                                img_path = os.path.join(frame_dir, img_file)
                                incident_map.setdefault((violation_label, date_obj), []).append(
                                    (img_path, f"{img_file}")
                                )
                except:
                    continue
    return incident_map

def display_incidents_ui():
    st.header("📂 Incidents Viewer")

    # Load once and reuse
    incident_data = load_incident_data()

    # Let user select violation type
    selected_violation = st.selectbox("Select Violation Type", list(VIOLATION_TYPES.keys()))

    # Get all dates available for this violation type
    available_dates = sorted(
        {date for (violation, date) in incident_data.keys() if violation == selected_violation},
        reverse=True
    )

    if not available_dates:
        st.warning("No incidents available for this violation type.")
        return

    # Date input
    selected_date = st.date_input("Select Date", value=available_dates[0],
                                  min_value=available_dates[-1], max_value=available_dates[0])

    # Fetch relevant frames
    selected_frames = incident_data.get((selected_violation, selected_date), [])
    st.markdown(f"### {selected_violation} incidents on {selected_date.strftime('%d-%m-%Y')}: **{len(selected_frames)}**")

    if not selected_frames:
        st.info("No incidents found for this date.")
        return

    # Paginate and show
    per_page = 9
    total_pages = math.ceil(len(selected_frames) / per_page)
    page = st.number_input("Page", min_value=1, max_value=total_pages, value=1, step=1)

    start = (page - 1) * per_page
    end = start + per_page
    current_frames = selected_frames[start:end]

    for row in range(0, len(current_frames), 3):
        cols = st.columns(3)
        for col, (img_path, caption) in zip(cols, current_frames[row:row + 4]):
            col.image(img_path, use_container_width=True, caption=caption)

if selected_tab == "Incidents":
    # ==== Graph of Incident Counts by Date ====
    incident_counts = {'no_helmet': Counter(), 'triple_riding': Counter(), 'no_parking': Counter()}

    incident_base = os.path.join(base_dir, 'incidents')
    for folder in os.listdir(incident_base):
        for incident_type in incident_counts:
            if folder.startswith(incident_type):
                date_str = folder.replace(f"{incident_type}_", "")
                frame_path = os.path.join(incident_base, folder, 'frames')
                if os.path.exists(frame_path):
                    count = len([f for f in os.listdir(frame_path) if f.endswith('.jpg')])
                    incident_counts[incident_type][date_str] += count

    # Flatten the data for Plotly
    data_for_graph = []
    for incident_type, date_counter in incident_counts.items():
        for date_str, count in sorted(date_counter.items()):
            data_for_graph.append({
                "Date": date_str,
                "Incident Type": incident_type.replace("_", " ").title(),
                "Count": count
            })

    if data_for_graph:
        fig = px.bar(data_for_graph, x="Date", y="Count", color="Incident Type",
                    barmode='group', title="📈 Incident Counts Over Time")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No incidents available for plotting.")
    
    display_incidents_ui()



# ==== Map Tab ====
if selected_tab == "Map":
    st.title("🌎 Real-time Map View")
    import streamlit.components.v1 as components
    map_url = "https://www.google.com/maps/embed?pb=!1m18!1m12!1m3!1d3888.9472910587865!2d77.63261757512184!3d12.911109287398782!2m3!1f0!2f0!3f0!3m2!1i1024!2i768!4f13.1!3m3!1m2!1s0x3bae15d1559e2651%3A0xb27e383e612faa20!2sLivNSense%20GreenOps%20Pvt%20Ltd!5e0!3m2!1sen!2sin!4v1744711532262!5m2!1sen!2sin"

    components.html(f"""
        <div style="border: 3px solid #4CAF50; border-radius: 15px; overflow: hidden; box-shadow: 0px 4px 8px rgba(0,0,0,0.1);">
            <iframe 
                src="{map_url}" 
                width="100%" 
                height="800" 
                style="border:0;" 
                allowfullscreen="" 
                loading="lazy">
            </iframe>
        </div>
    """, height=820)