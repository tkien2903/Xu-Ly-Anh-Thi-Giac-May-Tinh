# Phân biệt người / đồ vật và lưu nhãn học lại
import cv2, os, time, uuid
import numpy as np
from ultralytics import YOLO

# ======= CẤU HÌNH =======
CAMERA_INDEX = 0
CUSTOM_MODEL = "runs/detect/train6/weights/best.pt"
FALLBACK_MODEL = "yolov8n.pt"
CONF_THRESH = 0.35
MASK_PIXEL_THRESHOLD = 0.02
RESIZE_WIDTH = 960
LABEL_OUT_DIR = "feedback"
os.makedirs(os.path.join(LABEL_OUT_DIR, "images"), exist_ok=True)
os.makedirs(os.path.join(LABEL_OUT_DIR, "labels"), exist_ok=True)
os.makedirs(os.path.join(LABEL_OUT_DIR, "videos"), exist_ok=True)
# =========================


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
    frame = cv2.resize(frame, (width, int(h * (width / w))))  # resize theo width
    new_h = int(width * 9 / 16)
    cur_h = frame.shape[0]
    if cur_h > new_h:
        start = (cur_h - new_h) // 2
        frame = frame[start:start + new_h, :]
    elif cur_h < new_h:
        pad = (new_h - cur_h) // 2
        frame = cv2.copyMakeBorder(frame, pad, pad, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    return frame


def yolo_xyxy_to_norm(x1, y1, x2, y2, w, h):
    bw, bh = x2 - x1, y2 - y1
    cx, cy = x1 + bw / 2, y1 + bh / 2
    return cx / w, cy / h, bw / w, bh / h


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
            cid = 0 if label == "person" else 1  # 1 = object
            cx, cy, bw, bh = yolo_xyxy_to_norm(x1, y1, x2, y2, w, h)
            f.write(f"{cid} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")
    print(f"[+] Saved {img} and {txt}")


cap = cv2.VideoCapture(CAMERA_INDEX)
if not cap.isOpened():
    raise RuntimeError("Không mở được camera")

recording = False
video_writer = None
last_motion_time = 0
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_fps = 20

prev = time.time()
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    frame = safe_resize(frame, RESIZE_WIDTH)
    H, W = frame.shape[:2]

    # ====== PHÁT HIỆN CHUYỂN ĐỘNG ======
    fg = backSub.apply(frame)
    _, fg = cv2.threshold(fg, 200, 255, cv2.THRESH_BINARY)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, k, iterations=1)
    fg = cv2.dilate(fg, k, iterations=2)

    # ====== YOLO DETECTION ======
    res = model(frame, imgsz=640, conf=CONF_THRESH, verbose=False)
    dets = []
    for r in res:
        if r.boxes is None:
            continue
        for b in r.boxes:
            cid = int(b.cls[0])
            conf = float(b.conf[0])
            x1, y1, x2, y2 = map(int, b.xyxy[0].cpu().numpy())
            if x2 <= x1 or y2 <= y1:
                continue

            label = None
            if TWO_LABEL:
                # Trường hợp model 2 lớp: person / object
                n = class_names[cid].lower()
                if n in ("person", "object"):
                    label = n
            else:
                # Dùng YOLO COCO: 0 = person, còn lại coi là object
                if cid == 0:
                    label = "person"
                else:
                    label = "object"

            if not label or conf < CONF_THRESH:
                continue

            roi = fg[y1:y2, x1:x2]
            if roi.size == 0:
                continue
            motion = cv2.countNonZero(roi) / roi.size
            if motion < MASK_PIXEL_THRESHOLD:
                continue

            dets.append({"bbox": (x1, y1, x2, y2),
                         "cls": label,
                         "conf": conf,
                         "motion": motion})

    motion_detected = len(dets) > 0

    # ====== VẼ LÊN FRAME ======
    for d in dets:
        x1, y1, x2, y2 = d["bbox"]
        # person = xanh lá, object = xanh dương
        color = (0, 255, 0) if d["cls"] == "person" else (255, 0, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame,
                    f"{d['cls']} {d['conf'] * 100:.1f}%",
                    (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2)

    # >>> ĐẾM SỐ NGƯỜI TRONG KHUNG HÌNH
    person_count = sum(1 for d in dets if d["cls"] == "person")
    count_text = f"So nguoi: {person_count}"
    (tw, th), _ = cv2.getTextSize(count_text,
                                  cv2.FONT_HERSHEY_SIMPLEX,
                                  0.7,
                                  2)
    cv2.putText(frame,
                count_text,
                (W - tw - 10, H - 10),   # góc dưới bên phải
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2)

    # ====== GHI VIDEO KHI CÓ CHUYỂN ĐỘNG ======
    if motion_detected:
        last_motion_time = time.time()
        if not recording:
            uid = time.strftime("%Y%m%d_%H%M%S")
            video_path = os.path.join(LABEL_OUT_DIR, "videos", f"{uid}.mp4")
            video_writer = cv2.VideoWriter(video_path, fourcc, video_fps, (W, H))
            recording = True
            print(f"[REC] Bắt đầu ghi video: {video_path}")
    else:
        if recording and time.time() - last_motion_time > 3:
            recording = False
            video_writer.release()
            video_writer = None
            print("[STOP] Dừng ghi video.")

    if recording and video_writer is not None:
        video_writer.write(frame)

    fps = 1 / (time.time() - prev + 1e-5)
    prev = time.time()
    cv2.putText(frame,
                f"FPS: {fps:.1f}",
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2)

    if recording:
        cv2.putText(frame,
                    "Dang ghi hinh...",
                    (W - 180, 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2)

    cv2.putText(frame,
                "[S] save  [1] person  [2] object  [Q] quit",
                (10, H - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                1)

    cv2.imshow("Person/Object Detector", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s') and dets:
        save_sample(frame, dets)
    elif key == ord('1') and dets:
        save_sample(frame, dets, "person")
    elif key == ord('2') and dets:
        save_sample(frame, dets, "object")

if video_writer is not None:
    video_writer.release()
cap.release()
cv2.destroyAllWindows()
