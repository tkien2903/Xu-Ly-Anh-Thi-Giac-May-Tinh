# nhanbietnguoivsvat.py
import cv2, os, time, uuid
from ultralytics import YOLO

import smtplib
from email.message import EmailMessage
import cv2
import numpy as np

EMAIL_SENDER = "holehao2510@gmail.com"
EMAIL_APP_PASSWORD = "inol ptdq cjdd tawq"
EMAIL_RECEIVER = "holehao2510@gmail.com"


# ======= CẤU HÌNH =======
CAMERA_INDEX = 0
CUSTOM_MODEL = "runs/detect/train2/weights/best.pt"
FALLBACK_MODEL = "yolov8n.pt"
CONF_THRESH = 0.35
MASK_PIXEL_THRESHOLD = 0.02
RESIZE_WIDTH = 960
LABEL_OUT_DIR = "feedback"
os.makedirs(os.path.join(LABEL_OUT_DIR, "images"), exist_ok=True)
os.makedirs(os.path.join(LABEL_OUT_DIR, "labels"), exist_ok=True)
os.makedirs(os.path.join(LABEL_OUT_DIR, "videos"), exist_ok=True)
LOG_FILE = os.path.join(LABEL_OUT_DIR, "logs.txt")

def write_log_file(message):
    timestamp = time.strftime("%d/%m/%Y %H:%M:%S")
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"[{timestamp}] {message}\n")

def load_model():
    p = CUSTOM_MODEL if os.path.exists(CUSTOM_MODEL) else FALLBACK_MODEL
    model = YOLO(p)
    names = model.model.names if hasattr(model.model, "names") else model.names
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names.keys())]
    return model, names

model, class_names = load_model()

def is_two_label(names):
    if len(names) != 2:
        return False
    return {"person", "object"} == {n.lower() for n in names}

TWO_LABEL = is_two_label(class_names)
backSub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=25, detectShadows=True)

def safe_resize(frame, width=960):
    h, w = frame.shape[:2]
    frame = cv2.resize(frame, (width, int(h * (width / w))))
    new_h = int(width * 9 / 16)
    cur_h = frame.shape[0]
    if cur_h > new_h:
        start = (cur_h - new_h) // 2
        frame = frame[start:start + new_h, :]
    elif cur_h < new_h:
        pad = (new_h - cur_h) // 2
        frame = cv2.copyMakeBorder(frame, pad, pad, 0, 0, cv2.BORDER_CONSTANT, value=(0,0,0))
    return frame

def yolo_xyxy_to_norm(x1, y1, x2, y2, w, h):
    bw, bh = x2 - x1, y2 - y1
    cx, cy = x1 + bw/2, y1 + bh/2
    return cx/w, cy/h, bw/w, bh/h

def save_sample(frame, dets, override=None):
    h, w = frame.shape[:2]
    uid = uuid.uuid4().hex[:12]
    img = f"{uid}.jpg"
    txt = f"{uid}.txt"
    cv2.imwrite(os.path.join(LABEL_OUT_DIR, "images", img), frame)
    with open(os.path.join(LABEL_OUT_DIR, "labels", txt), "w", encoding="utf-8") as f:
        for d in dets:
            x1, y1, x2, y2 = d["bbox"]
            label = override or d["cls"]
            cid = 0 if label=="person" else 1
            cx, cy, bw, bh = yolo_xyxy_to_norm(x1, y1, x2, y2, w, h)
            f.write(f"{cid} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")
    print(f"[+] Saved {img} and {txt}")

def send_email_with_frame(frame):
    try:
        _, img_encoded = cv2.imencode(".jpg", frame)
        img_bytes = img_encoded.tobytes()

        msg = EmailMessage()
        msg["Subject"] = "⚠️ Phát hiện chuyển động"
        msg["From"] = EMAIL_SENDER
        msg["To"] = EMAIL_RECEIVER
        msg.set_content("Hệ thống vừa phát hiện có người di chuyển.")

        msg.add_attachment(
            img_bytes,
            maintype="image",
            subtype="jpeg",
            filename="snapshot.jpg"
        )

        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
            smtp.login(EMAIL_SENDER, EMAIL_APP_PASSWORD)
            smtp.send_message(msg)

        write_log_file("Đã gửi email cảnh báo kèm hình ảnh.")
        print("[MAIL] Sent frame!")

    except Exception as e:
        write_log_file(f"Lỗi gửi email: {e}")
        print("Mail error:", e)

# UI
STOP_FLAG = False
recording = False
video_writer = None
last_motion_time = 0
mail_sent = False
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_fps = 20
last_person = 0
last_object = 0

def detect(ui):
    global STOP_FLAG, recording, video_writer, last_motion_time, last_person, last_object, mail_sent
    STOP_FLAG = False
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        ui.add_log("Không mở được camera!", "error")
        write_log_file("Không mở được camera!")
        return

    prev = time.time()

    while not STOP_FLAG:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame,1)
        frame = safe_resize(frame, RESIZE_WIDTH)
        H,W = frame.shape[:2]

        # PHÁT HIỆN CHUYỂN ĐỘNG
        fg = backSub.apply(frame)
        _, fg = cv2.threshold(fg,200,255,cv2.THRESH_BINARY)
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, k, iterations=1)
        fg = cv2.dilate(fg,k,iterations=2)

        #  YOLO DETECTION
        res = model(frame, imgsz=640, conf=CONF_THRESH, verbose=False)
        dets = []
        for r in res:
            if r.boxes is None:
                continue
            for b in r.boxes:
                cid = int(b.cls[0])
                conf = float(b.conf[0])
                x1,y1,x2,y2 = map(int,b.xyxy[0].cpu().numpy())
                if x2<=x1 or y2<=y1: continue

                label=None
                if TWO_LABEL:
                    n = class_names[cid].lower()
                    if n in ("person","object"):
                        label=n
                else:
                    label="person" if cid==0 else "object"

                if not label or conf<CONF_THRESH: continue

                roi = fg[y1:y2, x1:x2]
                if roi.size==0: continue
                motion = cv2.countNonZero(roi)/roi.size
                if motion<MASK_PIXEL_THRESHOLD: continue

                dets.append({"bbox":(x1,y1,x2,y2),"cls":label,"conf":conf,"motion":motion})

        # VẼ LÊN FRAME
        for d in dets:
            x1,y1,x2,y2=d["bbox"]
            color = (0,255,0) if d["cls"]=="person" else (255,0,0)
            cv2.rectangle(frame,(x1,y1),(x2,y2),color,2)
            cv2.putText(frame,f"{d['cls']} {d['conf']*100:.1f}%",
                        (x1,y1-5),cv2.FONT_HERSHEY_SIMPLEX,0.6,color,2)

        now_text = time.strftime("%d/%m/%Y %H:%M:%S")
        cv2.putText(frame, now_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        person_count = sum(1 for d in dets if d["cls"]=="person")
        object_count = sum(1 for d in dets if d["cls"]=="object")

        # LOG THAY ĐỔI
        if person_count != last_person:
            msg = f"Phát hiện {person_count} người" if person_count>0 else "Không còn người"
            ui.add_log(msg,"warning" if person_count>0 else "info")
            write_log_file(msg)
            last_person = person_count

        if object_count != last_object:
            msg = f"Phát hiện {object_count} vật" if object_count>0 else "Không còn vật"
            ui.add_log(msg,"warning" if object_count>0 else "info")
            write_log_file(msg)
            last_object = object_count

        # UPDATE UI
        ui.update_video_frame(frame)
        ui.update_person_count(person_count)
        ui.update_object_count(object_count)

        ui.current_frame = frame
        ui.current_dets = dets

        # GHI VIDEO
        motion_detected = len(dets)>0
        person_detected = any(d["cls"] == "person" for d in dets)

        if motion_detected:
            last_motion_time=time.time()
            person_detected = any(d["cls"] == "person" for d in dets)

            if person_detected and not mail_sent:
                send_email_with_frame(frame)
                mail_sent = True
            if not recording:
                uid=time.strftime("%Y%m%d_%H%M%S")
                video_path=os.path.join(LABEL_OUT_DIR,"videos",f"{uid}.mp4")
                video_writer=cv2.VideoWriter(video_path,fourcc,video_fps,(W,H))
                recording=True
                ui.add_log(f"🔴 Bắt đầu ghi video: {video_path}","success")
                write_log_file(msg)
                ui.set_recording(True)
        else:
            if recording and time.time()-last_motion_time>3:
                recording=False
                video_writer.release()
                video_writer=None
                ui.add_log("🟡 Dừng ghi video.","warning")
                write_log_file(msg)
                ui.set_recording(False)
                mail_sent = False

        if recording and video_writer is not None:
            video_writer.write(frame)

    cap.release()
    if video_writer:
        video_writer.release()

    ui.video_label.config(image='')
    ui.current_frame = None
    ui.current_dets = []
