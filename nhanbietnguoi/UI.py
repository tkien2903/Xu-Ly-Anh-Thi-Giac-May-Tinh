# camera_ui.py
import tkinter as tk
from tkinter import scrolledtext
from PIL import Image, ImageTk
from datetime import datetime
import threading
import nhanbietnguoivsvat as det

class CameraUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Nhóm 10 - Camera giám sát chuyển động")
        self.root.geometry("1400x800")
        self.root.configure(bg='#2c3e50')

        self.camera_active=False
        self.detection_thread=None
        self.current_frame=None
        self.current_dets=[]
        self.setup_ui()

        # ===== BẮT PHÍM CHỤP ẢNH =====
        self.root.bind("<s>", lambda e: self.capture_all())
        self.root.bind("1", lambda e: self.capture_person())
        self.root.bind("2", lambda e: self.capture_object())

    def setup_ui(self):
        # === HEADER ===
        header_frame=tk.Frame(self.root,bg='#34495e',height=80)
        header_frame.pack(fill=tk.X,padx=10,pady=10)
        header_frame.pack_propagate(False)
        title_label=tk.Label(header_frame,text="NHÓM 10 NHẬN DIỆN CHUYỂN ĐỘNG",
                             font=('Arial',24,'bold'),bg='#34495e',fg='#ecf0f1')
        title_label.pack(side=tk.LEFT,padx=20,pady=20)

        self.btn_start=tk.Button(header_frame,text="▶ MỞ CAMERA",
                                 command=self.toggle_camera,font=('Arial',14,'bold'),
                                 bg='#27ae60',fg='white',width=15,height=2,cursor='hand2')
        self.btn_start.pack(side=tk.RIGHT,padx=20,pady=15)

        # === CONTENT ===
        content_frame=tk.Frame(self.root,bg='#2c3e50')
        content_frame.pack(fill=tk.BOTH,expand=True,padx=10,pady=5)

        # CAMERA
        camera_container=tk.Frame(content_frame,bg='#34495e',relief=tk.RAISED,bd=2)
        camera_container.pack(side=tk.LEFT,fill=tk.BOTH,expand=True,padx=(0,5))
        cam_label=tk.Label(camera_container,text="📹 CAMERA",font=('Arial',12,'bold'),
                           bg='#34495e',fg='#ecf0f1')
        cam_label.pack(pady=5)
        self.video_label=tk.Label(camera_container,bg='black')
        self.video_label.pack(fill=tk.BOTH,expand=True,padx=10,pady=10)

        # CHỤP ẢNH
        self.hd_label=tk.Label(camera_container,
                               text="Hướng dẫn: [S] Chụp tất cả, [1] Chụp người, [2] Chụp vật",
                               font=('Arial',10,'bold'),bg='#34495e',fg='yellow')
        self.hd_label.pack(pady=5)

        # LOG
        log_container=tk.Frame(content_frame,bg='#34495e',width=400,relief=tk.RAISED,bd=2)
        log_container.pack(side=tk.RIGHT,fill=tk.BOTH,padx=(5,0))
        log_container.pack_propagate(False)
        log_header=tk.Label(log_container,text="📋 NHẬT KÝ HOẠT ĐỘNG",font=('Arial',12,'bold'),
                            bg='#34495e',fg='#ecf0f1')
        log_header.pack(pady=5)

        # STATS
        stats_frame=tk.Frame(log_container,bg='#2c3e50')
        stats_frame.pack(fill=tk.X,padx=10,pady=5)
        self.label_person_count=tk.Label(stats_frame,text="👤 Người: 0",font=('Arial',11,'bold'),
                                         bg='#27ae60',fg='white',padx=10,pady=5,relief=tk.RAISED)
        self.label_person_count.pack(side=tk.LEFT,padx=5)
        self.label_object_count=tk.Label(stats_frame,text="📦 Vật: 0",font=('Arial',11,'bold'),
                                         bg='#3498db',fg='white',padx=10,pady=5,relief=tk.RAISED)
        self.label_object_count.pack(side=tk.LEFT,padx=5)
        self.label_status=tk.Label(stats_frame,text="⚫ Offline",font=('Arial',11,'bold'),
                                   bg='#95a5a6',fg='white',padx=10,pady=5,relief=tk.RAISED)
        self.label_status.pack(side=tk.RIGHT,padx=5)
        self.label_recording=tk.Label(log_container,text="",font=('Arial',10,'bold'),
                                      bg='#34495e',fg='#e74c3c')
        self.label_recording.pack(pady=2)

        self.log_text=scrolledtext.ScrolledText(log_container,font=('Consolas',9),
                                                bg='#1c2833',fg='#ecf0f1',insertbackground='white',
                                                wrap=tk.WORD,relief=tk.FLAT,state=tk.NORMAL)
        self.log_text.pack(fill=tk.BOTH,expand=True,padx=10,pady=10)
        self.log_text.tag_config('info',foreground='#3498db')
        self.log_text.tag_config('success',foreground='#27ae60')
        self.log_text.tag_config('warning',foreground='#f39c12')
        self.log_text.tag_config('error',foreground='#e74c3c')
        self.log_text.tag_config('time',foreground='#95a5a6')

    # ===== LOG =====
    def add_log(self,message,level="info"):
        timestamp=datetime.now().strftime("%H:%M:%S")
        self.log_text.insert(tk.END,f"[{timestamp}] ",'time')
        self.log_text.insert(tk.END,f"{message}\n",level)
        self.log_text.see(tk.END)

    # ===== UPDATE UI =====
    def update_person_count(self,count):
        self.label_person_count.config(text=f"👤 Người: {count}")
    def update_object_count(self,count):
        self.label_object_count.config(text=f"📦 Vật: {count}")
    def set_status(self,status,color):
        self.label_status.config(text=status,bg=color)
    def set_recording(self,is_recording):
        if is_recording:
            self.label_recording.config(text="🔴 ĐANG GHI HÌNH")
        else:
            self.label_recording.config(text="")

    # ===== HIỂN THỊ CAMERA =====
    def update_video_frame(self,frame):
        frame_rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        img=Image.fromarray(frame_rgb)
        img=img.resize((960,540),Image.Resampling.LANCZOS)
        imgtk=ImageTk.PhotoImage(image=img)
        self.video_label.imgtk=imgtk
        self.video_label.configure(image=imgtk)

    # ===== TOGGLE CAMERA =====
    def toggle_camera(self):
        self.camera_active=not self.camera_active
        if self.camera_active:
            self.btn_start.config(text="⏸ TẮT CAMERA",bg='#e74c3c')
            self.set_status("🔴 Online",'#27ae60')
            self.add_log("Camera đã bật","success")
            det.STOP_FLAG=False
            self.detection_thread=threading.Thread(target=det.detect,args=(self,),daemon=True)
            self.detection_thread.start()
        else:
            det.STOP_FLAG=True
            self.btn_start.config(text="▶ MỞ CAMERA",bg='#27ae60')
            self.set_status("⚫ Offline",'#95a5a6')
            self.add_log("Camera đã tắt","warning")
            self.video_label.config(image='')
            self.current_frame=None
            self.current_dets=[]

    # ===== CHỤP ẢNH QUA PHÍM =====
    def capture_all(self):
        if self.current_frame is not None and self.current_dets:
            det.save_sample(self.current_frame,self.current_dets)
            self.add_log("📸 Đã chụp hình tất cả","success")
    def capture_person(self):
        if self.current_frame is not None and self.current_dets:
            det.save_sample(self.current_frame,self.current_dets,"person")
            self.add_log("📸 Đã chụp hình người","success")
    def capture_object(self):
        if self.current_frame is not None and self.current_dets:
            det.save_sample(self.current_frame,self.current_dets,"object")
            self.add_log("📸 Đã chụp hình vật","success")

if __name__=="__main__":
    import cv2
    root=tk.Tk()
    ui=CameraUI(root)
    ui.add_log("Hệ thống khởi động","info")
    root.mainloop()
