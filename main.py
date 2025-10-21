import cv2
import numpy as np
import tkinter as tk
from tkinter import messagebox as mbox
from keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
from collections import Counter, deque
from PIL import Image, ImageTk
from itertools import islice
import winsound
import time
from datetime import datetime
from cryptography.fernet import Fernet
import os
import csv

print("Program Started...")

# ---------------- AES Encryption Setup ---------------- #
KEY_FILE = "log_key.key"

def get_or_create_key():
    if os.path.exists(KEY_FILE):
        return open(KEY_FILE, "rb").read()
    key = Fernet.generate_key()
    open(KEY_FILE, "wb").write(key)
    return key

FERNET = Fernet(get_or_create_key())

def write_encrypted_csv_row(filename, row):
    line = ",".join(map(str, row)) + "\n"
    token = FERNET.encrypt(line.encode())
    with open(filename, "ab") as f:
        f.write(token + b"\n")

def read_encrypted_csv(filename):
    out = []
    if not os.path.exists(filename):
        return out
    with open(filename, "rb") as f:
        for token in f:
            token = token.strip()
            if not token:
                continue
            try:
                plain = FERNET.decrypt(token).decode()
                out.append(plain)
            except Exception:
                continue
    return out

# ---------------- Decrypted CSV Save & Reset ---------------- #
def save_and_reset_logs(enc_file="predictions_log.enc", out_file="emotion_log.csv"):
    logs = read_encrypted_csv(enc_file)
    if not logs:
        print("No logs to save.")
    else:
        with open(out_file, "w", newline='') as f:
            writer = csv.writer(f)
            for line in logs:
                writer.writerow(line.strip().split(","))
        print(f"Session logs saved to {out_file}")
    open(enc_file, "wb").close()
    print("Encrypted log reset for fresh start.")

# ---------------- Alert CSV ---------------- #
ALERT_LOG_FILE = "alerts_log.csv"
def write_alert_log(timestamp, person_id, emotion, alert_type):
    with open(ALERT_LOG_FILE, "a", newline='') as f:
        writer = csv.writer(f)
        writer.writerow([timestamp, person_id, emotion, alert_type])

# ---------------- Clear previous log on startup ---------------- #
if os.path.exists("predictions_log.enc"):
    open("predictions_log.enc", "wb").close()
    print("Previous encrypted log cleared. Starting fresh session.")

# ---------------- Face + Emotion Setup ---------------- #
face_classifier = cv2.CascadeClassifier(r'haarcascade_frontalface_default.xml')
classifier = load_model(r'model.h5')
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# ---------------- Tkinter GUI Setup ---------------- #
root = tk.Tk()
root.title("Emotion Recognition Dashboard")
root.geometry("1500x900")
root.configure(bg="white")

# ---------------- Top Banner (Color Changed) ---------------- #
banner_label = tk.Label(root, text="🧠 Emotion Recognition Dashboard",
                        font=("Helvetica", 24, "bold"),
                        bg="#007ACC", fg="white", pady=10)
banner_label.grid(row=0, column=0, columnspan=2, sticky="we", padx=5, pady=5)

# ---------------- Alert Box ---------------- #
alert_label = tk.Label(root, text="No Alerts", font=("Helvetica", 18, "bold"),
                       bg="white", fg="green", relief="ridge", bd=3, pady=5)
alert_label.grid(row=1, column=0, columnspan=2, sticky="we", padx=10, pady=5)

# ---------------- Video & Charts Setup ---------------- #
video_label = tk.Label(root, bg="white")
video_label.grid(row=2, column=0, padx=10, pady=10, sticky="nsew")

fig, (ax_bar, ax_line, ax_pie) = plt.subplots(3, 1, figsize=(6, 10))
fig.tight_layout(pad=3)
chart_canvas = FigureCanvasTkAgg(fig, master=root)
chart_canvas.get_tk_widget().grid(row=2, column=1, padx=10, pady=10, sticky="nsew")

# ---------------- Data Structures ---------------- #
person_counts = {}
person_history = {}
last_alert_time = {}
ALERT_THRESHOLD = 0.6
ALERT_HISTORY_LENGTH = 10
SUSPICIOUS_EMOTIONS = ["Angry", "Fear", "Sad"]
ALERT_COOLDOWN = 5  # per-person cooldown (seconds)
STARTUP_COOLDOWN = 5  # global cooldown at startup

program_start_time = time.time()

# ---------------- Camera Setup ---------------- #
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# ---------------- Alert Functions ---------------- #
def flash_alert():
    current_color = alert_label.cget("fg")
    next_color = "red" if current_color != "red" else "white"
    alert_label.configure(fg=next_color)
    root.after(500, flash_alert)

def check_alert(person_id, current_emotion):
    now = time.time()

    # Global cooldown at program start
    if now - program_start_time < STARTUP_COOLDOWN:
        return

    if person_id not in person_history:
        person_history[person_id] = deque(maxlen=ALERT_HISTORY_LENGTH)
    if person_id not in last_alert_time:
        last_alert_time[person_id] = 0

    person_history[person_id].append(current_emotion)
    recent = list(person_history[person_id])
    count_suspicious = sum(e in SUSPICIOUS_EMOTIONS for e in recent)
    recent_ratio = count_suspicious / len(recent)

    # Check cooldown per person
    if (recent_ratio >= ALERT_THRESHOLD and current_emotion in SUSPICIOUS_EMOTIONS
            and now - last_alert_time[person_id] >= ALERT_COOLDOWN):
        last_alert_time[person_id] = now
        alert_label.configure(text=f"⚠ {person_id}: {current_emotion} detected!", fg="red")
        winsound.Beep(1000, 500)
        flash_alert()
        mbox.showwarning("ALERT", f"{person_id}: Suspicious Emotion Detected: {current_emotion}")
        write_alert_log(datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f"), person_id, current_emotion, "Suspicious")
    else:
        alert_label.configure(text="No Alerts", fg="green")

# ---------------- Video + Emotion Update ---------------- #
def update_frame():
    ret, frame = cap.read()
    if not ret:
        print("Camera not available!")
        root.after(1000, update_frame)
        return

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)

    for idx, (x, y, w, h) in enumerate(faces, start=1):
        person_id = f"Person_{idx}"
        roi_gray = gray[y:y+h, x:x+w]
        roi_gray = cv2.resize(roi_gray, (48,48), interpolation=cv2.INTER_AREA)
        roi = roi_gray.astype('float') / 255.0
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)

        prediction = classifier.predict(roi, verbose=0)[0]
        label = emotion_labels[prediction.argmax()]

        if person_id not in person_counts:
            person_counts[person_id] = Counter()
        person_counts[person_id][label] += 1

        timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
        write_encrypted_csv_row("predictions_log.enc",
                                [timestamp_str, person_id, label,
                                 "ALERT" if label in SUSPICIOUS_EMOTIONS else "NORMAL"])

        check_alert(person_id, label)

        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,255), 2)
        cv2.putText(frame, f"{person_id}: {label}", (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

    # Update charts
    ax_bar.clear()
    ax_line.clear()
    ax_pie.clear()
    if person_counts:
        for person_id, counts in person_counts.items():
            ax_bar.bar(counts.keys(), counts.values(), label=person_id)
            ax_line.plot(list(range(len(person_history.get(person_id, [])))),
                         [emotion_labels.index(e)
                          for e in person_history.get(person_id, [])],
                         marker="o", linestyle="-", label=person_id)
        ax_bar.set_title("Emotion Count per Person")
        ax_bar.set_ylabel("Frequency")
        ax_line.set_title("Emotion Trend Over Time per Person")
        ax_line.set_ylabel("Emotion Index")
        ax_line.set_xlabel("Time Step")
        ax_line.set_yticks(range(len(emotion_labels)))
        ax_line.set_yticklabels(emotion_labels, rotation=30)
        combined_counts = Counter()
        for counts in person_counts.values():
            combined_counts.update(counts)
        ax_pie.pie(combined_counts.values(), labels=combined_counts.keys(),
                   autopct="%1.1f%%", startangle=90)
        ax_pie.set_title("Combined Emotion Distribution")

    chart_canvas.draw()

    cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = cv2.resize(cv2image, (500,400))
    imgtk = ImageTk.PhotoImage(image=Image.fromarray(img))
    video_label.imgtk = imgtk
    video_label.configure(image=imgtk)

    root.after(10, update_frame)

# ---------------- Buttons Frame ---------------- #
buttons_frame = tk.Frame(root, bg="white")
buttons_frame.grid(row=3, column=0, columnspan=2, pady=10)

log_button = tk.Button(buttons_frame, text="Save Logs CSV", command=save_and_reset_logs,
                       font=("Helvetica", 12, "bold"), bg="#4CAF50", fg="white", padx=10, pady=5)
log_button.pack(side="left", padx=20)

exit_button = tk.Button(buttons_frame, text="Exit", command=root.destroy,
                        font=("Helvetica", 12, "bold"), bg="#f44336", fg="white", padx=10, pady=5)
exit_button.pack(side="left", padx=20)

# ---------------- Configure Grid Weights ---------------- #
root.grid_rowconfigure(2, weight=1)
root.grid_columnconfigure(0, weight=1)
root.grid_columnconfigure(1, weight=1)

# ---------------- Save logs on window close ---------------- #
def on_exit():
    save_and_reset_logs()
    root.destroy()

root.protocol("WM_DELETE_WINDOW", on_exit)

# ---------------- Run ---------------- #
update_frame()
print("Starting Tkinter mainloop...")
root.mainloop()

cap.release()
cv2.destroyAllWindows()
