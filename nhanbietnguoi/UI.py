import tkinter as tk
from tkinter import scrolledtext
from PIL import Image, ImageTk
from datetime import datetime
import threading

class CameraUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Nhóm 10 - Nhận diện chuyển động")
        self.root.geometry("1400x800")
        self.root.configure(bg='#2c3e50')
        
        self.camera_active = False
        self.detection_thread = None
        self.setup_ui()
        
    def setup_ui(self):

        header_frame = tk.Frame(self.root, bg='#34495e', height=80)
        header_frame.pack(fill=tk.X, padx=10, pady=10)
        header_frame.pack_propagate(False)
        
        title_label = tk.Label(
            header_frame,
            text="NHÓM 10 NHẬN DIỆN CHUYỂN ĐỘNG",
            font=('Arial', 24, 'bold'),
            bg='#34495e',
            fg='#ecf0f1'
        )
        title_label.pack(side=tk.LEFT, padx=20, pady=20)
        
        self.btn_start = tk.Button(
            header_frame,
            text="▶ MỞ CAMERA",
            command=self.toggle_camera,
            font=('Arial', 14, 'bold'),
            bg='#27ae60',
            fg='white',
            width=15,
            height=2,
            cursor='hand2',
            relief=tk.RAISED,
            bd=3
        )
        self.btn_start.pack(side=tk.RIGHT, padx=20, pady=15)
        
        content_frame = tk.Frame(self.root, bg='#2c3e50')
        content_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        camera_container = tk.Frame(content_frame, bg='#34495e', relief=tk.RAISED, bd=2)
        camera_container.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        cam_label = tk.Label(
            camera_container,
            text="📹 CAMERA",
            font=('Arial', 12, 'bold'),
            bg='#34495e',
            fg='#ecf0f1'
        )
        cam_label.pack(pady=5)
        
        self.video_label = tk.Label(camera_container, bg='black')
        self.video_label.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        log_container = tk.Frame(content_frame, bg='#34495e', width=400, relief=tk.RAISED, bd=2)
        log_container.pack(side=tk.RIGHT, fill=tk.BOTH, padx=(5, 0))
        log_container.pack_propagate(False)
        
        log_header = tk.Label(
            log_container,
            text="📋 NHẬT KÝ HOẠT ĐỘNG",
            font=('Arial', 12, 'bold'),
            bg='#34495e',
            fg='#ecf0f1'
        )
        log_header.pack(pady=5)
        
        stats_frame = tk.Frame(log_container, bg='#2c3e50')
        stats_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.label_person_count = tk.Label(
            stats_frame,
            text="👤 Người: 0",
            font=('Arial', 11, 'bold'),
            bg='#27ae60',
            fg='white',
            padx=10,
            pady=5,
            relief=tk.RAISED
        )
        self.label_person_count.pack(side=tk.LEFT, padx=5)
        
        self.label_object_count = tk.Label(
            stats_frame,
            text="📦 Vật: 0",
            font=('Arial', 11, 'bold'),
            bg='#3498db',
            fg='white',
            padx=10,
            pady=5,
            relief=tk.RAISED
        )
        self.label_object_count.pack(side=tk.LEFT, padx=5)
        
        self.label_status = tk.Label(
            stats_frame,
            text="⚫ Offline",
            font=('Arial', 11, 'bold'),
            bg='#95a5a6',
            fg='white',
            padx=10,
            pady=5,
            relief=tk.RAISED
        )
        self.label_status.pack(side=tk.RIGHT, padx=5)
        
        self.label_recording = tk.Label(
            log_container,
            text="",
            font=('Arial', 10, 'bold'),
            bg='#34495e',
            fg='#e74c3c'
        )
        self.label_recording.pack(pady=2)
        
        self.log_text = scrolledtext.ScrolledText(
            log_container,
            font=('Consolas', 9),
            bg='#1c2833',
            fg='#ecf0f1',
            insertbackground='white',
            wrap=tk.WORD,
            relief=tk.FLAT,
            state=tk.NORMAL
        )
        self.log_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.log_text.tag_config('info', foreground='#3498db')
        self.log_text.tag_config('success', foreground='#27ae60')
        self.log_text.tag_config('warning', foreground='#f39c12')
        self.log_text.tag_config('error', foreground='#e74c3c')
        self.log_text.tag_config('time', foreground='#95a5a6')
        
    def add_log(self, message, level="info"):
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_text.insert(tk.END, f"[{timestamp}] ", 'time')
        self.log_text.insert(tk.END, f"{message}\n", level)
        self.log_text.see(tk.END)
    
    def update_person_count(self, count):
        self.label_person_count.config(text=f"👤 Người: {count}")
    
    def update_object_count(self, count):
        self.label_object_count.config(text=f"📦 Vật: {count}")
    
    def set_status(self, status, color):
        self.label_status.config(text=status, bg=color)
    
    def set_recording(self, is_recording):
        if is_recording:
            self.label_recording.config(text="🔴 ĐANG GHI HÌNH")
        else:
            self.label_recording.config(text="")
    
    def update_video_frame(self, frame):
        import cv2
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        img = img.resize((960, 540), Image.Resampling.LANCZOS)
        imgtk = ImageTk.PhotoImage(image=img)
        self.video_label.imgtk = imgtk
        self.video_label.configure(image=imgtk)
    
    def toggle_camera(self):

        self.camera_active = not self.camera_active

        if self.camera_active:

            # import đúng file nhận dạng của bạn
            import nhanbietnguoivsvat as det

            self.btn_start.config(text="⏸ TẮT CAMERA", bg='#e74c3c')
            self.set_status("🔴 Online", '#27ae60')
            self.add_log("Camera đã bật", "success")

            det.STOP_FLAG = False
            self.detection_thread = threading.Thread(
                target=det.detect,
                args=(self,),
                daemon=True
            )
            self.detection_thread.start()

        else:
            import nhanbietnguoivsvat as det
            det.STOP_FLAG = True

            self.btn_start.config(text="▶ MỞ CAMERA", bg='#27ae60')
            self.set_status("⚫ Offline", '#95a5a6')
            self.add_log("Camera đã tắt", "warning")

            self.video_label.config(image='')


if __name__ == "__main__":
    root = tk.Tk()
    ui = CameraUI(root)
    ui.add_log("Hệ thống khởi động", "info")
    root.mainloop()
