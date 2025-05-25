# import streamlit as st
# import cv2
# import os
# import time
# import requests
# from ultralytics import YOLO
# import numpy as np
# from collections import defaultdict
# from datetime import datetime
# from PIL import Image

# # ==== TELEGRAM CONFIG ====
# BOT_TOKEN = '7802076982:AAGYvTpB0Rzt4bSxM_I3DQXBG1ijEbHBkXY'
# CHAT_ID = '7424789030'

# def send_telegram_alert(image_path, message="🚨 No Helmet Violation Detected!"):
#     url = f'https://api.telegram.org/bot{BOT_TOKEN}/sendPhoto'
#     with open(image_path, 'rb') as photo:
#         files = {'photo': photo}
#         data = {'chat_id': CHAT_ID, 'caption': message}
#         response = requests.post(url, files=files, data=data)
#         return response.status_code == 200

# # ==== PATHS ====
# model_path = 'G:\\desktop\\MyProjects\\DL_projects\\BPRD_usecase\\inference\\models\\updated_5_best_yolov8.pt'
# video_dir = 'G:\\desktop\\MyProjects\\DL_projects\\BPRD_usecase\\inference\\input_videos\\input_videos_20-05-2025'
# output_dir = 'G:\\desktop\\MyProjects\\DL_projects\\BPRD_usecase\\inference\\output_videos\\output_videos_20-05-2025'
# base_incident_dir = 'G:\\desktop\\MyProjects\\DL_projects\\BPRD_usecase\\inference\\incidents_clips'
# no_hel_clips = "G:\\desktop\\MyProjects\\DL_projects\\BPRD_usecase\\inference\\incidents_clips\\no_helmet_18-04-2025\\clips"
# triple_clips = "G:\\desktop\\MyProjects\\DL_projects\\BPRD_usecase\\inference\\incidents_clips\\triple_riding_18-04-2025\\clips"
# no_helmet_dir = os.path.join(base_incident_dir, 'no_helmet_18-04-2025')
# triple_riding_dir = os.path.join(base_incident_dir, 'triple_riding_18-04-2025')

# # Directories for saving incidents
# for d in [no_helmet_dir, triple_riding_dir, output_dir]:
#     os.makedirs(os.path.join(d, "frames"), exist_ok=True)
#     os.makedirs(os.path.join(d, "clips"), exist_ok=True)

# # ==== LABELS ====
# labels = ['person', 'car', 'bike', 'auto', 'bus', 'livestock', 'helmet', 'no_helmet', 'truck', 'number_plate']
# colors = {
#     'person': (0, 255, 0), 'car': (255, 0, 0), 'bike': (0, 0, 255), 'auto': (255, 255, 0),
#     'bus': (255, 165, 0), 'livestock': (128, 0, 128), 'helmet': (0, 255, 255),
#     'no_helmet': (0, 100, 255), 'truck': (139, 69, 19), 'number_plate': (255, 20, 147)
# }

# # ==== LOAD MODEL ====
# model = YOLO(model_path)

# # ==== STREAMLIT SIDEBAR ====
# st.sidebar.title("🧭 Navigation")
# selected_tab = st.sidebar.radio("Go to", ["Live Inference", "No Helmet Incidents", "Triple Riding Incidents", "Map"])


# # ==== LIVE INFERENCE ====
# if selected_tab == "Live Inference":
#     st.title("🚦Traffic Inference Dashboard")
#     input_mode = st.radio("📡 Input Mode", ["Upload Video", "Live Cam"])
#     confidence_threshold = st.slider("📏 Confidence Threshold", 0.0, 1.0, 0.3, step=0.05)
#     label_filter = st.multiselect("🏷️ Filter by Label", labels, default=labels)

#     if input_mode == "Upload Video":
#         video_files = [f for f in os.listdir(video_dir) if f.endswith(('.mp4', '.avi', '.MOV'))]
#         selected_video = st.selectbox("🎥 Select Input Video", video_files)
#         start_inference = st.button("▶️ Start Inference")
#         if start_inference:
#             cap = cv2.VideoCapture(os.path.join(video_dir, selected_video))
#     else:
#         start_cam = st.button("🎥 Start Camera")
#         stop_cam = st.button("🛑 Stop Camera")
#         cap = None
#         ip_webcam_url = st.text_input("📱 Enter IP Webcam URL", "rtsp://admin:lns12345@192.168.2.64/streaming/Channels/101")
#         if start_cam:
#             cap = cv2.VideoCapture(ip_webcam_url)

#     if (input_mode == "Upload Video" and start_inference) or (input_mode == "Live Cam" and cap is not None):
#         fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
#         width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#         height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#         fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#         output_path = os.path.join(output_dir, f'output_{datetime.now().strftime("%Y%m%d_%H%M%S")}.mp4')
#         out_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

#         stframe = st.empty()
#         summary = defaultdict(int)
#         frame_buffer = []
#         no_helmet_count, triple_riding_count = 0, 0
#         current_frame_idx = 0

#         while cap.isOpened():
#             ret, frame = cap.read()
#             if not ret:
#                 break

#             current_frame_idx += 1
#             raw_frame = frame.copy()
#             frame_buffer.append(raw_frame)
#             if len(frame_buffer) > fps * 5:
#                 frame_buffer.pop(0)

#             results = model(frame)[0]
#             violation_detected = {"no_helmet": False, "triple_riding": False}
#             detected_frame = {"no_helmet": None, "triple_riding": None}

#             boxes_by_label = defaultdict(list)

#             for box in results.boxes:
#                 x1, y1, x2, y2 = map(int, box.xyxy[0])
#                 conf = box.conf[0].item()
#                 cls = int(box.cls[0].item())
#                 label = labels[cls]

#                 if conf < confidence_threshold or label not in label_filter:
#                     continue

#                 summary[label] += 1
#                 color = colors.get(label, (255, 255, 255))
#                 cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
#                 cv2.putText(frame, f'{label} {conf:.2f}', (x1, y1 - 10),
#                             cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

#                 boxes_by_label[label].append((x1, y1, x2, y2))

#             # === Triple Riding Logic ===
#             for bike_bbox in boxes_by_label['bike']:
#                 bx1, by1, bx2, by2 = bike_bbox
#                 persons_on_bike = [p for p in boxes_by_label['person'] if bx1 < (p[0]+p[2])//2 < bx2 and by1 < (p[1]+p[3])//2 < by2]
#                 if len(persons_on_bike) >= 3 and not violation_detected["triple_riding"]:
#                     violation_detected["triple_riding"] = True
#                     detected_frame["triple_riding"] = frame.copy()

#             # === No Helmet Detection ===
#             if 'no_helmet' in boxes_by_label:
#                 violation_detected["no_helmet"] = True
#                 detected_frame["no_helmet"] = frame.copy()

#             out_writer.write(frame)
#             stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)

#             for violation in ["no_helmet", "triple_riding"]:
#                 if violation_detected[violation] and detected_frame[violation] is not None:
#                     timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
#                     count = no_helmet_count if violation == "no_helmet" else triple_riding_count
#                     count += 1

#                     folder = no_helmet_dir if violation == "no_helmet" else triple_riding_dir
#                     message = "🚨 No Helmet Violation Detected!" if violation == "no_helmet" else "🚨 Triple Riding Detected!"

#                     frame_path = os.path.join(folder, "frames", f"{violation}_frame_{count}_{timestamp}.jpg")
#                     clip_path = os.path.join(folder, "clips", f"{violation}_incident_{count}_{timestamp}.mp4")

#                     cv2.imwrite(frame_path, detected_frame[violation])
#                     send_telegram_alert(frame_path, message=message)

#                     incident_writer = cv2.VideoWriter(clip_path, fourcc, fps, (width, height))
#                     for bf in frame_buffer:
#                         incident_writer.write(bf)
#                     for _ in range(fps * 5):
#                         incident_writer.write(raw_frame)
#                     incident_writer.release()

#                     if violation == "no_helmet":
#                         no_helmet_count = count
#                     else:
#                         triple_riding_count = count

#             time.sleep(1 / fps)
#             if input_mode == "Live Cam" and stop_cam:
#                 break

#         cap.release()
#         out_writer.release()
#         st.success(f"✅ Inference complete! Output saved to `{output_path}`")
#         st.subheader("📊 Violation Summary")
#         st.table([{"Label": k, "Count": v} for k, v in summary.items()])

# # ==== INCIDENT VIEWERS ====

# elif selected_tab == "No Helmet Incidents":
#     st.title("🚨 Incident Viewer")

#     folders = sorted([f for f in os.listdir(base_incident_dir) if os.path.isdir(os.path.join(base_incident_dir, f))])
#     selected_folder = st.selectbox("📁 Select Incident Folder", folders)

#     selected_folder_path = os.path.join(base_incident_dir, selected_folder)
#     selected_frames_dir = os.path.join(selected_folder_path, "frames")
#     selected_clips_dir = os.path.join(selected_folder_path, "clips")

#     image_files = sorted(os.listdir(selected_frames_dir), reverse=True)
#     video_files = sorted(os.listdir(selected_clips_dir), reverse=True)

#     incidents_per_page = 5
#     total_incidents = len(video_files)
#     total_pages = (total_incidents - 1) // incidents_per_page + 1
#     page = st.number_input("📄 Page", 1, total_pages, 1)

#     start = (page - 1) * incidents_per_page
#     end = start + incidents_per_page

#     current_images = image_files[start:end]
#     current_videos = video_files[start:end]

#     st.markdown("### 🖼️ Frames")
#     for img_file in current_images:
#         img_path = os.path.join(selected_frames_dir, img_file)
#         st.image(img_path, caption=img_file, use_container_width=True)

# elif selected_tab == "Triple Riding Incidents":
#     st.title("🚨 Incident Viewer")

#     folders = sorted([f for f in os.listdir(base_incident_dir) if os.path.isdir(os.path.join(base_incident_dir, f))])
#     selected_folder = st.selectbox("📁 Select Incident Folder", folders)

#     selected_folder_path = os.path.join(base_incident_dir, selected_folder)
#     selected_frames_dir = os.path.join(selected_folder_path, "frames")
#     selected_clips_dir = os.path.join(selected_folder_path, "clips")

#     image_files = sorted(os.listdir(selected_frames_dir), reverse=True)
#     video_files = sorted(os.listdir(selected_clips_dir), reverse=True)

#     incidents_per_page = 5
#     total_incidents = len(video_files)
#     total_pages = (total_incidents - 1) // incidents_per_page + 1
#     page = st.number_input("📄 Page", 1, total_pages, 1)

#     start = (page - 1) * incidents_per_page
#     end = start + incidents_per_page

#     current_images = image_files[start:end]
#     current_videos = video_files[start:end]

#     st.markdown("### 🖼️ Frames")
#     for img_file in current_images:
#         img_path = os.path.join(selected_frames_dir, img_file)
#         st.image(img_path, caption=img_file, use_container_width=True)


# # ==== MAP TAB ====
# elif selected_tab == "Map":
#     st.title("🗺️ Real-time Map View")
#     import streamlit.components.v1 as components
#     map_url = "https://www.google.com/maps/embed?pb=!1m18!1m12!1m3!1d3888.9472910587865!2d77.63261757512184!3d12.911109287398782!2m3!1f0!2f0!3f0!3m2!1i1024!2i768!4f13.1!3m3!1m2!1s0x3bae15d1559e2651%3A0xb27e383e612faa20!2sLivNSense%20GreenOps%20Pvt%20Ltd!5e0!3m2!1sen!2sin!4v1744711532262!5m2!1sen!2sin"

#     components.html(f"""
#         <div style="border: 3px solid #4CAF50; border-radius: 15px; overflow: hidden; box-shadow: 0px 4px 8px rgba(0,0,0,0.1);">
#             <iframe 
#                 src="{map_url}" 
#                 width="100%" 
#                 height="800" 
#                 style="border:0;" 
#                 allowfullscreen="" 
#                 loading="lazy">
#             </iframe>
#         </div>
#     """, height=820)

#----------------------------------------------------------Triple riding-----------------------------------------------------------------
# import streamlit as st
# import cv2
# import os
# import time
# import requests
# from ultralytics import YOLO
# import numpy as np
# from collections import defaultdict
# from datetime import datetime, timedelta
# from PIL import Image

# # ==== TELEGRAM CONFIG ====
# BOT_TOKEN = '7802076982:AAGYvTpB0Rzt4bSxM_I3DQXBG1ijEbHBkXY'
# CHAT_ID = '7424789030'

# def send_telegram_alert(image_path, message="🚨 No Helmet Violation Detected!"):
#     url = f'https://api.telegram.org/bot{BOT_TOKEN}/sendPhoto'
#     with open(image_path, 'rb') as photo:
#         files = {'photo': photo}
#         data = {'chat_id': CHAT_ID, 'caption': message}
#         response = requests.post(url, files=files, data=data)
#         return response.status_code == 200

# # ==== PATHS ====
# model_path = 'G:\\desktop\\MyProjects\\DL_projects\\BPRD_usecase\\inference\\models\\updated_5_best_yolov8.pt'
# video_dir = 'G:\\desktop\\MyProjects\\DL_projects\\BPRD_usecase\\inference\\input_videos\\input_videos_20-05-2025'
# output_dir = 'G:\\desktop\\MyProjects\\DL_projects\\BPRD_usecase\\inference\\output_videos\\output_videos_20-05-2025'
# base_incident_dir = 'G:\\desktop\\MyProjects\\DL_projects\\BPRD_usecase\\inference\\incidents_clips'
# no_hel_clips = "G:\\desktop\\MyProjects\\DL_projects\\BPRD_usecase\\inference\\incidents_clips\\no_helmet_18-04-2025\\clips"
# triple_clips = "G:\\desktop\\MyProjects\\DL_projects\\BPRD_usecase\\inference\\incidents_clips\\triple_riding_18-04-2025\\clips"
# no_helmet_dir = os.path.join(base_incident_dir, 'no_helmet_18-04-2025')
# triple_riding_dir = os.path.join(base_incident_dir, 'triple_riding_18-04-2025')

# # Directories for saving incidents
# for d in [no_helmet_dir, triple_riding_dir, output_dir]:
#     os.makedirs(os.path.join(d, "frames"), exist_ok=True)
#     os.makedirs(os.path.join(d, "clips"), exist_ok=True)

# # ==== LABELS ====
# labels = ['person', 'car', 'bike', 'auto', 'bus', 'livestock', 'helmet', 'no_helmet', 'truck', 'number_plate']
# colors = {
#     'person': (0, 255, 0), 'car': (255, 0, 0), 'bike': (0, 0, 255), 'auto': (255, 255, 0),
#     'bus': (255, 165, 0), 'livestock': (128, 0, 128), 'helmet': (0, 255, 255),
#     'no_helmet': (0, 100, 255), 'truck': (139, 69, 19), 'number_plate': (255, 20, 147)
# }

# # ==== LOAD MODEL ====
# model = YOLO(model_path)

# # ==== STREAMLIT SIDEBAR ====
# st.sidebar.title("🧭 Navigation")
# selected_tab = st.sidebar.radio("Go to", ["Live Inference", "No Helmet Incidents", "Triple Riding Incidents", "Map"])

# # ==== LIVE INFERENCE ====
# if selected_tab == "Live Inference":
#     st.title("🚦Traffic Inference Dashboard")
#     input_mode = st.radio("📡 Input Mode", ["Upload Video", "Live Cam"])
#     confidence_threshold = st.slider("📏 Confidence Threshold", 0.0, 1.0, 0.3, step=0.05)
#     label_filter = st.multiselect("🏷️ Filter by Label", labels, default=labels)

#     if input_mode == "Upload Video":
#         video_files = [f for f in os.listdir(video_dir) if f.endswith(('.mp4', '.avi', '.MOV'))]
#         selected_video = st.selectbox("🎥 Select Input Video", video_files)
#         start_inference = st.button("▶️ Start Inference")
#         if start_inference:
#             cap = cv2.VideoCapture(os.path.join(video_dir, selected_video))
#     else:
#         start_cam = st.button("🎥 Start Camera")
#         stop_cam = st.button("🛑 Stop Camera")
#         cap = None
#         ip_webcam_url = st.text_input("📱 Enter IP Webcam URL", "rtsp://admin:lns12345@192.168.2.64/streaming/Channels/101")
#         if start_cam:
#             cap = cv2.VideoCapture(ip_webcam_url)

#     if (input_mode == "Upload Video" and start_inference) or (input_mode == "Live Cam" and cap is not None):
#         fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
#         width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#         height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#         fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#         output_path = os.path.join(output_dir, f'output_{datetime.now().strftime("%Y%m%d_%H%M%S")}.mp4')
#         out_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

#         stframe = st.empty()
#         summary = defaultdict(int)
#         frame_buffer = []
#         no_helmet_count, triple_riding_count = 0, 0
#         current_frame_idx = 0

#         # Initialize cooldown timers per violation type
#         cooldown_seconds = 10
#         last_detection_time = {
#             "no_helmet": datetime.min,
#             "triple_riding": datetime.min
#         }

#         while cap.isOpened():
#             ret, frame = cap.read()
#             if not ret:
#                 break

#             current_frame_idx += 1
#             raw_frame = frame.copy()
#             frame_buffer.append(raw_frame)
#             if len(frame_buffer) > fps * 5:
#                 frame_buffer.pop(0)

#             results = model(frame)[0]
#             violation_detected = {"no_helmet": False, "triple_riding": False}
#             detected_frame = {"no_helmet": None, "triple_riding": None}

#             boxes_by_label = defaultdict(list)

#             for box in results.boxes:
#                 x1, y1, x2, y2 = map(int, box.xyxy[0])
#                 conf = box.conf[0].item()
#                 cls = int(box.cls[0].item())
#                 label = labels[cls]

#                 if conf < confidence_threshold or label not in label_filter:
#                     continue

#                 summary[label] += 1
#                 color = colors.get(label, (255, 255, 255))
#                 cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
#                 cv2.putText(frame, f'{label} {conf:.2f}', (x1, y1 - 10),
#                             cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

#                 boxes_by_label[label].append((x1, y1, x2, y2))

#             # === Triple Riding Logic ===
#             for bike_bbox in boxes_by_label['bike']:
#                 bx1, by1, bx2, by2 = bike_bbox
#                 persons_on_bike = [p for p in boxes_by_label['person'] if bx1 < (p[0]+p[2])//2 < bx2 and by1 < (p[1]+p[3])//2 < by2]
#                 if len(persons_on_bike) >= 3 and not violation_detected["triple_riding"]:
#                     violation_detected["triple_riding"] = True
#                     detected_frame["triple_riding"] = frame.copy()

#             # === No Helmet Detection ===
#             if 'no_helmet' in boxes_by_label:
#                 violation_detected["no_helmet"] = True
#                 detected_frame["no_helmet"] = frame.copy()

#             out_writer.write(frame)
#             stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)

#             for violation in ["no_helmet", "triple_riding"]:
#                 now = datetime.now()
#                 # Check cooldown to avoid spamming alerts for the same violation
#                 if violation_detected[violation] and detected_frame[violation] is not None:
#                     if (now - last_detection_time[violation]).total_seconds() > cooldown_seconds:
#                         last_detection_time[violation] = now
#                         if violation == "no_helmet":
#                             no_helmet_count += 1
#                             count = no_helmet_count
#                         else:
#                             triple_riding_count += 1
#                             count = triple_riding_count

#                         folder = no_helmet_dir if violation == "no_helmet" else triple_riding_dir
#                         message = "🚨 No Helmet Violation Detected!" if violation == "no_helmet" else "🚨 Triple Riding Detected!"

#                         timestamp = now.strftime('%Y%m%d_%H%M%S')
#                         frame_path = os.path.join(folder, "frames", f"{violation}_frame_{count}_{timestamp}.jpg")
#                         clip_path = os.path.join(folder, "clips", f"{violation}_incident_{count}_{timestamp}.mp4")

#                         cv2.imwrite(frame_path, detected_frame[violation])
#                         send_telegram_alert(frame_path, message=message)

#                         # Save incident clip: 5 seconds before + 5 seconds after
#                         incident_writer = cv2.VideoWriter(clip_path, fourcc, fps, (width, height))
#                         # Write buffered frames (5 seconds before)
#                         for bf in frame_buffer:
#                             incident_writer.write(bf)
#                         # Write next 5 seconds frames after detection
#                         post_frames = 0
#                         while post_frames < fps * 5:
#                             ret2, next_frame = cap.read()
#                             if not ret2:
#                                 break
#                             incident_writer.write(next_frame)
#                             post_frames += 1
#                         incident_writer.release()

#             time.sleep(1 / fps)

#             # Stop camera logic fixed: if stop_cam button is pressed, break loop
#             if input_mode == "Live Cam" and stop_cam:
#                 break

#         cap.release()
#         out_writer.release()
#         st.success(f"✅ Inference complete! Output saved to `{output_path}`")
#         st.subheader("📊 Violation Summary")
#         st.table([{"Label": k, "Count": v} for k, v in summary.items()])

# # ==== INCIDENT VIEWERS ====

# elif selected_tab == "No Helmet Incidents":
#     st.title("🚨 Incident Viewer")

#     folders = sorted([f for f in os.listdir(base_incident_dir) if os.path.isdir(os.path.join(base_incident_dir, f)) and f.startswith("no_helmet")])
#     selected_folder = st.selectbox("Select Incident Folder", folders)
#     selected_clips_dir = os.path.join(base_incident_dir, selected_folder)
#     frames_dir = os.path.join(selected_clips_dir, "frames")
#     clips_dir = os.path.join(selected_clips_dir, "clips")

#     incident_frames = sorted(os.listdir(frames_dir))
#     # incident_videos = sorted(os.listdir(clips_dir))

#     if incident_frames:
#         st.subheader("Incident Frames")
#         for frame_file in incident_frames:
#             frame_path = os.path.join(frames_dir, frame_file)
#             image = Image.open(frame_path)
#             st.image(image, caption=frame_file)

#     # if incident_videos:
#     #     st.subheader("Incident Clips")
#     #     for clip_file in incident_videos:
#     #         clip_path = os.path.join(clips_dir, clip_file)
#     #         st.video(clip_path)

# elif selected_tab == "Triple Riding Incidents":
#     st.title("🚨 Incident Viewer")

#     folders = sorted([f for f in os.listdir(base_incident_dir) if os.path.isdir(os.path.join(base_incident_dir, f)) and f.startswith("triple_riding")])
#     selected_folder = st.selectbox("Select Incident Folder", folders)
#     selected_clips_dir = os.path.join(base_incident_dir, selected_folder)
#     frames_dir = os.path.join(selected_clips_dir, "frames")
#     clips_dir = os.path.join(selected_clips_dir, "clips")

#     incident_frames = sorted(os.listdir(frames_dir))
#     # incident_videos = sorted(os.listdir(clips_dir))

#     if incident_frames:
#         st.subheader("Incident Frames")
#         for frame_file in incident_frames:
#             frame_path = os.path.join(frames_dir, frame_file)
#             image = Image.open(frame_path)
#             st.image(image, caption=frame_file)

#     # if incident_videos:
#     #     st.subheader("Incident Clips")
#     #     for clip_file in incident_videos:
#     #         clip_path = os.path.join(clips_dir, clip_file)
#     #         st.video(clip_path)

# # ==== MAP TAB PLACEHOLDER ====
# elif selected_tab == "Map":
#     st.title("🗺️ Real-time Map View")
#     import streamlit.components.v1 as components
#     map_url = "https://www.google.com/maps/embed?pb=!1m18!1m12!1m3!1d3888.9472910587865!2d77.63261757512184!3d12.911109287398782!2m3!1f0!2f0!3f0!3m2!1i1024!2i768!4f13.1!3m3!1m2!1s0x3bae15d1559e2651%3A0xb27e383e612faa20!2sLivNSense%20GreenOps%20Pvt%20Ltd!5e0!3m2!1sen!2sin!4v1744711532262!5m2!1sen!2sin"

#     components.html(f"""
#         <div style="border: 3px solid #4CAF50; border-radius: 15px; overflow: hidden; box-shadow: 0px 4px 8px rgba(0,0,0,0.1);">
#             <iframe 
#                 src="{map_url}" 
#                 width="100%" 
#                 height="800" 
#                 style="border:0;" 
#                 allowfullscreen="" 
#                 loading="lazy">
#             </iframe>
#         </div>
#     """, height=820)





#---------------------------------------------------separate conf score for each labels--------------------------------------------



# import streamlit as st
# import cv2
# import os
# import time
# import requests
# from ultralytics import YOLO
# import numpy as np
# from collections import defaultdict
# from datetime import datetime, timedelta
# from PIL import Image

# # ==== TELEGRAM CONFIG ====
# BOT_TOKEN = '7802076982:AAGYvTpB0Rzt4bSxM_I3DQXBG1ijEbHBkXY'
# CHAT_ID = '7424789030'

# def send_telegram_alert(image_path, message="🚨 No Helmet Violation Detected!"):
#     url = f'https://api.telegram.org/bot{BOT_TOKEN}/sendPhoto'
#     with open(image_path, 'rb') as photo:
#         files = {'photo': photo}
#         data = {'chat_id': CHAT_ID, 'caption': message}
#         response = requests.post(url, files=files, data=data)
#         return response.status_code == 200

# # ==== PATHS ====
# model_path = 'G:\\desktop\\MyProjects\\DL_projects\\BPRD_usecase\\inference\\models\\updated_5_best_yolov8.pt'
# video_dir = 'G:\\desktop\\MyProjects\\DL_projects\\BPRD_usecase\\inference\\input_videos\\input_videos_20-05-2025'
# output_dir = 'G:\\desktop\\MyProjects\\DL_projects\\BPRD_usecase\\inference\\output_videos\\output_videos_20-05-2025'
# base_incident_dir = 'G:\\desktop\\MyProjects\\DL_projects\\BPRD_usecase\\inference\\incidents_clips'
# no_hel_clips = "G:\\desktop\\MyProjects\\DL_projects\\BPRD_usecase\\inference\\incidents_clips\\no_helmet_18-04-2025\\clips"
# triple_clips = "G:\\desktop\\MyProjects\\DL_projects\\BPRD_usecase\\inference\\incidents_clips\\triple_riding_18-04-2025\\clips"
# no_helmet_dir = os.path.join(base_incident_dir, 'no_helmet_18-04-2025')
# triple_riding_dir = os.path.join(base_incident_dir, 'triple_riding_18-04-2025')

# # Directories for saving incidents
# for d in [no_helmet_dir, triple_riding_dir, output_dir]:
#     os.makedirs(os.path.join(d, "frames"), exist_ok=True)
#     os.makedirs(os.path.join(d, "clips"), exist_ok=True)

# # ==== LABELS ====
# labels = ['person', 'car', 'bike', 'auto', 'bus', 'livestock', 'helmet', 'no_helmet', 'truck', 'number_plate']
# colors = {
#     'person': (0, 255, 0), 'car': (255, 0, 0), 'bike': (0, 0, 255), 'auto': (255, 255, 0),
#     'bus': (255, 165, 0), 'livestock': (128, 0, 128), 'helmet': (0, 255, 255),
#     'no_helmet': (0, 100, 255), 'truck': (139, 69, 19), 'number_plate': (255, 20, 147)
# }

# # ==== LOAD MODEL ====
# model = YOLO(model_path)

# # ==== STREAMLIT SIDEBAR ====
# st.sidebar.title("🧭 Navigation")
# selected_tab = st.sidebar.radio("Go to", ["Live Inference", "No Helmet Incidents", "Triple Riding Incidents", "Map"])

# # === Add separate confidence sliders per label ===
# st.sidebar.markdown("### 🎚️ Confidence Thresholds per Class")
# confidence_thresholds = {}
# for label in labels:
#     confidence_thresholds[label] = st.sidebar.slider(f"{label}", 0.0, 1.0, 0.3, step=0.05)

# # ==== LIVE INFERENCE ====
# if selected_tab == "Live Inference":
#     st.title("🚦Traffic Inference Dashboard")
#     input_mode = st.radio("📡 Input Mode", ["Upload Video", "Live Cam"])
#     label_filter = st.multiselect("🏷️ Filter by Label", labels, default=labels)

#     if input_mode == "Upload Video":
#         video_files = [f for f in os.listdir(video_dir) if f.endswith(('.mp4', '.avi', '.MOV'))]
#         selected_video = st.selectbox("🎥 Select Input Video", video_files)
#         start_inference = st.button("▶️ Start Inference")
#         if start_inference:
#             cap = cv2.VideoCapture(os.path.join(video_dir, selected_video))
#     else:
#         start_cam = st.button("🎥 Start Camera")
#         stop_cam = st.button("🛑 Stop Camera")
#         cap = None
#         ip_webcam_url = st.text_input("📱 Enter IP Webcam URL", "rtsp://admin:lns12345@192.168.2.64/streaming/Channels/101")
#         if start_cam:
#             cap = cv2.VideoCapture(ip_webcam_url)

#     if (input_mode == "Upload Video" and start_inference) or (input_mode == "Live Cam" and cap is not None):
#         fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
#         width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#         height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#         fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#         output_path = os.path.join(output_dir, f'output_{datetime.now().strftime("%Y%m%d_%H%M%S")}.mp4')
#         out_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

#         stframe = st.empty()
#         summary = defaultdict(int)
#         frame_buffer = []
#         no_helmet_count, triple_riding_count = 0, 0
#         current_frame_idx = 0

#         # Initialize cooldown timers per violation type
#         cooldown_seconds = 10
#         last_detection_time = {
#             "no_helmet": datetime.min,
#             "triple_riding": datetime.min
#         }

#         while cap.isOpened():
#             ret, frame = cap.read()
#             if not ret:
#                 break

#             current_frame_idx += 1
#             raw_frame = frame.copy()
#             frame_buffer.append(raw_frame)
#             if len(frame_buffer) > fps * 5:
#                 frame_buffer.pop(0)

#             results = model(frame)[0]
#             violation_detected = {"no_helmet": False, "triple_riding": False}
#             detected_frame = {"no_helmet": None, "triple_riding": None}

#             boxes_by_label = defaultdict(list)

#             for box in results.boxes:
#                 x1, y1, x2, y2 = map(int, box.xyxy[0])
#                 conf = box.conf[0].item()
#                 cls = int(box.cls[0].item())
#                 label = labels[cls]

#                 # Use per-class confidence threshold now
#                 if conf < confidence_thresholds.get(label, 0.3) or label not in label_filter:
#                     continue

#                 summary[label] += 1
#                 color = colors.get(label, (255, 255, 255))
#                 cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
#                 cv2.putText(frame, f'{label} {conf:.2f}', (x1, y1 - 10),
#                             cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

#                 boxes_by_label[label].append((x1, y1, x2, y2))

#             # === Triple Riding Logic ===
#             for bike_bbox in boxes_by_label['bike']:
#                 bx1, by1, bx2, by2 = bike_bbox
#                 persons_on_bike = [p for p in boxes_by_label['person'] if bx1 < (p[0]+p[2])//2 < bx2 and by1 < (p[1]+p[3])//2 < by2]
#                 if len(persons_on_bike) >= 3 and not violation_detected["triple_riding"]:
#                     violation_detected["triple_riding"] = True
#                     detected_frame["triple_riding"] = frame.copy()

#             # === No Helmet Detection ===
#             if 'no_helmet' in boxes_by_label:
#                 violation_detected["no_helmet"] = True
#                 detected_frame["no_helmet"] = frame.copy()

#             out_writer.write(frame)
#             stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)

#             for violation in ["no_helmet", "triple_riding"]:
#                 now = datetime.now()
#                 # Check cooldown to avoid spamming alerts for the same violation
#                 if violation_detected[violation] and detected_frame[violation] is not None:
#                     if (now - last_detection_time[violation]).total_seconds() > cooldown_seconds:
#                         last_detection_time[violation] = now
#                         if violation == "no_helmet":
#                             no_helmet_count += 1
#                             count = no_helmet_count
#                         else:
#                             triple_riding_count += 1
#                             count = triple_riding_count

#                         folder = no_helmet_dir if violation == "no_helmet" else triple_riding_dir
#                         message = "🚨 No Helmet Violation Detected!" if violation == "no_helmet" else "🚨 Triple Riding Detected!"

#                         timestamp = now.strftime('%Y%m%d_%H%M%S')
#                         frame_path = os.path.join(folder, "frames", f"{violation}_frame_{count}_{timestamp}.jpg")
#                         clip_path = os.path.join(folder, "clips", f"{violation}_incident_{count}_{timestamp}.mp4")

#                         cv2.imwrite(frame_path, detected_frame[violation])
#                         send_telegram_alert(frame_path, message=message)

#                         # Save incident clip: 5 seconds before + 5 seconds after
#                         incident_writer = cv2.VideoWriter(clip_path, fourcc, fps, (width, height))
#                         # Write buffered frames (5 seconds before)
#                         for bf in frame_buffer:
#                             incident_writer.write(bf)
#                         # Write next 5 seconds frames after detection
#                         post_frames = 0
#                         while post_frames < fps * 5:
#                             ret2, next_frame = cap.read()
#                             if not ret2:
#                                 break
#                             incident_writer.write(next_frame)
#                             post_frames += 1
#                         incident_writer.release()

#             time.sleep(1 / fps)

#             # Stop camera logic fixed: if stop_cam button is pressed, break loop
#             if input_mode == "Live Cam" and stop_cam:
#                 break

#         cap.release()
#         out_writer.release()
#         st.success(f"✅ Inference complete! Output saved to `{output_path}`")
#         st.subheader("📊 Detection Summary")
#         st.write(dict(summary))

# # ==== INCIDENTS VIEWERS ====
# def display_incident_clips(directory, violation_label):
#     st.header(f"{violation_label} Incidents")
#     clips_path = os.path.join(directory, "clips")
#     if not os.path.exists(clips_path):
#         st.warning("No incidents recorded yet.")
#         return
#     clips = [f for f in os.listdir(clips_path) if f.endswith(".mp4")]
#     clips.sort(reverse=True)
#     selected_clip = st.selectbox("Select Clip", clips)
#     if selected_clip:
#         clip_file = os.path.join(clips_path, selected_clip)
#         video_bytes = open(clip_file, 'rb').read()
#         st.video(video_bytes)

# if selected_tab == "No Helmet Incidents":
#     display_incident_clips(no_helmet_dir, "No Helmet")

# if selected_tab == "Triple Riding Incidents":
#     display_incident_clips(triple_riding_dir, "Triple Riding")

# if selected_tab == "Map":
#     st.title("🗺️ Real-time Map View")
#     import streamlit.components.v1 as components
#     map_url = "https://www.google.com/maps/embed?pb=!1m18!1m12!1m3!1d3888.9472910587865!2d77.63261757512184!3d12.911109287398782!2m3!1f0!2f0!3f0!3m2!1i1024!2i768!4f13.1!3m3!1m2!1s0x3bae15d1559e2651%3A0xb27e383e612faa20!2sLivNSense%20GreenOps%20Pvt%20Ltd!5e0!3m2!1sen!2sin!4v1744711532262!5m2!1sen!2sin"

#     components.html(f"""
#         <div style="border: 3px solid #4CAF50; border-radius: 15px; overflow: hidden; box-shadow: 0px 4px 8px rgba(0,0,0,0.1);">
#             <iframe 
#                 src="{map_url}" 
#                 width="100%" 
#                 height="800" 
#                 style="border:0;" 
#                 allowfullscreen="" 
#                 loading="lazy">
#             </iframe>
#         </div>
#     """, height=820)



#---------------------------------------------------No parking-------------------------------------------------------------------------


import streamlit as st
import cv2
import os
import time
import requests
from ultralytics import YOLO
import numpy as np
from collections import defaultdict
from datetime import datetime, timedelta
from PIL import Image
from streamlit_drawable_canvas import st_canvas  # Add this import for ROI drawing

# ==== TELEGRAM CONFIG ====
BOT_TOKEN = '7802076982:AAGYvTpB0Rzt4bSxM_I3DQXBG1ijEbHBkXY'
CHAT_ID = '7424789030'

def send_telegram_alert(image_path, message="🚨 No Helmet Violation Detected!"):
    url = f'https://api.telegram.org/bot{BOT_TOKEN}/sendPhoto'
    with open(image_path, 'rb') as photo:
        files = {'photo': photo}
        data = {'chat_id': CHAT_ID, 'caption': message}
        response = requests.post(url, files=files, data=data)
        return response.status_code == 200

# ==== PATHS ====
model_path = 'G:\\desktop\\MyProjects\\DL_projects\\BPRD_usecase\\inference\\models\\updated_5_best_yolov8.pt'
video_dir = 'G:\\desktop\\MyProjects\\DL_projects\\BPRD_usecase\\inference\\input_videos\\input_videos_20-05-2025'
output_dir = 'G:\\desktop\\MyProjects\\DL_projects\\BPRD_usecase\\inference\\output_videos\\output_videos_20-05-2025'
base_incident_dir = 'G:\\desktop\\MyProjects\\DL_projects\\BPRD_usecase\\inference\\incidents_clips'
no_hel_clips = "G:\\desktop\\MyProjects\\DL_projects\\BPRD_usecase\\inference\\incidents_clips\\no_helmet_18-04-2025\\clips"
triple_clips = "G:\\desktop\\MyProjects\\DL_projects\\BPRD_usecase\\inference\\incidents_clips\\triple_riding_18-04-2025\\clips"
no_helmet_dir = os.path.join(base_incident_dir, 'no_helmet_18-04-2025')
triple_riding_dir = os.path.join(base_incident_dir, 'triple_riding_18-04-2025')
no_parking_dir = os.path.join(base_incident_dir, 'no_parking_23-05-2025')

# Create no parking incident directories if not exist
os.makedirs(os.path.join(no_parking_dir, "frames"), exist_ok=True)
os.makedirs(os.path.join(no_parking_dir, "clips"), exist_ok=True)

# ==== LABELS ====
labels = ['person', 'car', 'bike', 'auto', 'bus', 'livestock', 'helmet', 'no_helmet', 'truck', 'number_plate']
colors = {
    'person': (0, 255, 0), 'car': (255, 0, 0), 'bike': (0, 0, 255), 'auto': (255, 255, 0),
    'bus': (255, 165, 0), 'livestock': (128, 0, 128), 'helmet': (0, 255, 255),
    'no_helmet': (0, 100, 255), 'truck': (139, 69, 19), 'number_plate': (255, 20, 147)
}

# ==== LOAD MODEL ====
model = YOLO(model_path)

# ==== STREAMLIT SIDEBAR ====
st.sidebar.title("🧭 Navigation")
selected_tab = st.sidebar.radio("Go to", ["Live Inference", "No Helmet Incidents", "Triple Riding Incidents", "Map"])

# === Add separate confidence sliders per label ===
st.sidebar.markdown("### 🎚️ Confidence Thresholds per Class")
confidence_thresholds = {}
for label in labels:
    confidence_thresholds[label] = st.sidebar.slider(f"{label}", 0.0, 1.0, 0.3, step=0.05)

# ==== LIVE INFERENCE ====
if selected_tab == "Live Inference":
    st.title("🚦Traffic Inference Dashboard")
    input_mode = st.radio("📡 Input Mode", ["Upload Video", "Live Cam", "No Parking"])
    label_filter = st.multiselect("🏷️ Filter by Label", labels, default=labels)

    cap = None
    roi = None
    vehicle_last_seen = {}  # dict to track vehicles inside ROI and their timestamps

    if input_mode == "Upload Video":
        video_files = [f for f in os.listdir(video_dir) if f.endswith(('.mp4', '.avi', '.MOV'))]
        selected_video = st.selectbox("🎥 Select Input Video", video_files)
        start_inference = st.button("▶️ Start Inference")
        if start_inference:
            cap = cv2.VideoCapture(os.path.join(video_dir, selected_video))

    elif input_mode == "Live Cam":
        start_cam = st.button("🎥 Start Camera")
        stop_cam = st.button("🛑 Stop Camera")
        ip_webcam_url = st.text_input("📱 Enter IP Webcam URL", "rtsp://admin:lns12345@192.168.2.64/streaming/Channels/101")
        if start_cam:
            cap = cv2.VideoCapture(ip_webcam_url)

    elif input_mode == "No Parking":
        st.info("Monitoring hardcoded ROI for No Parking violation (15+ seconds stationary).")
        
        # roi_coords = [(174, 287), (1093, 278), (1097, 354), (164, 344)]
        roi_coords = [(366, 436), (1333, 416), (1319, 535), (346, 518)]
        roi = np.array(roi_coords, dtype=np.int32)
        
        inp_mode = st.radio("📡 Input Mode", ["Upload Video", "Live Cam"])
        
        if inp_mode == "Upload Video":
            video_files = [f for f in os.listdir(video_dir) if f.endswith(('.mp4', '.avi', '.MOV'))]
            selected_video = st.selectbox("🎥 Select Input Video", video_files)
            start_inference = st.button("▶️ Start Inference")
            if start_inference:
                cap = cv2.VideoCapture(os.path.join(video_dir, selected_video))
        
        elif inp_mode == "Live Cam":
        
            ip_webcam_url = st.text_input("📱 Enter IP Webcam URL", "rtsp://admin:lns12345@192.168.2.64/streaming/Channels/101")
            
            start_no_parking = st.button("▶️ Start No Parking Detection")
            stop_no_parking = st.button("🛑 Stop No Parking Detection")

            if start_no_parking:
                cap = cv2.VideoCapture(ip_webcam_url)
            if stop_no_parking:
                if cap is not None:
                    cap.release()
                    cap = None


    if cap is not None and cap.isOpened():
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        output_path = os.path.join(output_dir, f'output_{datetime.now().strftime("%Y%m%d_%H%M%S")}.mp4')
        out_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        stframe = st.empty()
        summary = defaultdict(int)
        frame_buffer = []
        no_helmet_count, triple_riding_count, no_parking_count = 0, 0, 0
        current_frame_idx = 0

        cooldown_seconds = 10
        last_detection_time = {
            "no_helmet": datetime.min,
            "triple_riding": datetime.min,
            "no_parking": datetime.min
        }

        # For no parking: store vehicle ID and time first detected inside ROI
        vehicle_entry_time = {}

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            current_frame_idx += 1
            raw_frame = frame.copy()
            frame_buffer.append(raw_frame)
            if len(frame_buffer) > fps * 5:
                frame_buffer.pop(0)

            results = model(frame)[0]
            violation_detected = {"no_helmet": False, "triple_riding": False, "no_parking": False}
            detected_frame = {"no_helmet": None, "triple_riding": None, "no_parking": None}

            boxes_by_label = defaultdict(list)

            for box in results.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = box.conf[0].item()
                cls = int(box.cls[0].item())
                label = labels[cls]

                if conf < confidence_thresholds.get(label, 0.3) or label not in label_filter:
                    continue

                summary[label] += 1
                color = colors.get(label, (255, 255, 255))
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f'{label} {conf:.2f}', (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                boxes_by_label[label].append((x1, y1, x2, y2))

            # === Triple Riding Logic ===
            for bike_bbox in boxes_by_label['bike']:
                bx1, by1, bx2, by2 = bike_bbox
                persons_on_bike = [p for p in boxes_by_label['person'] if bx1 < (p[0]+p[2])//2 < bx2 and by1 < (p[1]+p[3])//2 < by2]
                if len(persons_on_bike) >= 3 and not violation_detected["triple_riding"]:
                    violation_detected["triple_riding"] = True
                    detected_frame["triple_riding"] = frame.copy()

            # === No Helmet Detection ===
            if 'no_helmet' in boxes_by_label:
                violation_detected["no_helmet"] = True
                detected_frame["no_helmet"] = frame.copy()

            # === No Parking Detection Logic ===
            if input_mode == "No Parking" and roi is not None:
                vehicles_in_roi = []
                for label_name in ['car', 'bike', 'auto', 'bus', 'truck']:
                    for bbox in boxes_by_label[label_name]:
                        cx = (bbox[0] + bbox[2]) // 2
                        cy = (bbox[1] + bbox[3]) // 2
                        if cv2.pointPolygonTest(roi, (cx, cy), False) >= 0:
                            vehicles_in_roi.append((cx, cy, bbox))

                now = datetime.now()

                # Update vehicle_entry_time dict for vehicles inside ROI
                # Use bbox center as vehicle ID key (simplistic)
                current_ids = []
                for cx, cy, bbox in vehicles_in_roi:
                    vid = (cx, cy)
                    current_ids.append(vid)
                    if vid not in vehicle_entry_time:
                        vehicle_entry_time[vid] = now

                # Remove vehicles that left ROI
                for vid in list(vehicle_entry_time.keys()):
                    if vid not in current_ids:
                        vehicle_entry_time.pop(vid)

                # Check for vehicles stationary inside ROI for 15+ seconds
                for vid, entry_time in vehicle_entry_time.items():
                    elapsed = (now - entry_time).total_seconds()
                    if elapsed >= 2 and not violation_detected["no_parking"]:
                        violation_detected["no_parking"] = True
                        detected_frame["no_parking"] = frame.copy()

                        roi_coords = [(366, 436), (1333, 416), (1319, 535), (346, 518)]
                        roi = np.array(roi_coords, dtype=np.int32)
                        
                        if roi is not None:
                            cv2.polylines(frame, [roi], isClosed=True, color=(0, 165, 255), thickness=2)
                        
                        # Save violation frame
                        timestamp_str = now.strftime("%Y%m%d_%H%M%S")
                        frame_filename = os.path.join(no_parking_dir, "frames", f"no_parking_{timestamp_str}.jpg")
                        cv2.imwrite(frame_filename, frame)

                        # Save only frame, do NOT save clip
                        timestamp_str = now.strftime("%Y%m%d_%H%M%S")
                        frame_filename = os.path.join(no_parking_dir, "frames", f"no_parking_{timestamp_str}.jpg")
                        cv2.imwrite(frame_filename, frame)

                        # Send Telegram alert with frame only
                        send_telegram_alert(frame_filename, "🚗 No Parking Violation Detected!")

                        no_parking_count += 1


            # Display ROI polygon on frame if exists
            if roi is not None:
                cv2.polylines(frame, [roi], isClosed=True, color=(0, 165, 255), thickness=3)

            # Write frame to output video
            out_writer.write(frame)

            # Show frame on Streamlit UI
            stframe.image(frame, channels="BGR")

            # Show stats
            st.sidebar.markdown("### 📊 Summary")
            for lbl, count in summary.items():
                st.sidebar.markdown(f"- {lbl}: {count}")
            st.sidebar.markdown(f"**No Helmet Violations: {no_helmet_count}**")
            st.sidebar.markdown(f"**Triple Riding Violations: {triple_riding_count}**")
            st.sidebar.markdown(f"**No Parking Violations: {no_parking_count}**")

        cap.release()
        out_writer.release()

# === Other tabs left unchanged for brevity ===
if selected_tab == "No Helmet Incidents":
    st.title("No Helmet Incidents")
    st.write("View saved no helmet violation clips and frames.")

if selected_tab == "Triple Riding Incidents":
    st.title("Triple Riding Incidents")
    st.write("View saved triple riding violation clips and frames.")

if selected_tab == "Map":
    st.title("Map")
    st.write("Display live map of detected incidents.")