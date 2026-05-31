from flask import Flask, render_template, Response
import cv2
from ultralytics import YOLO
import serial
import time

app = Flask(__name__)

# ======================
# LOAD MODEL AI
# ======================
model = YOLO("best2.pt")

# ======================
# CONNECT ESP32
# ======================
try:
    esp32 = serial.Serial()
    esp32.port = 'COM6'
    esp32.baudrate = 115200
    esp32.timeout = 1
    
    # 🔥 MATIKAN SINYAL RESET OTOMATIS
    esp32.setDTR(False)
    esp32.setRTS(False)
    
    esp32.open()
    time.sleep(2)
    print("✅ ESP32 Connected!")
except Exception as e:
    esp32 = None
    print("⚠️ ESP32 NOT CONNECTED:")
    print(e)

# ======================
# CAMERA + AI + ESP32
# ======================
def generate_frames():
    cap = cv2.VideoCapture(1)

    last_sent_time = 0
    cooldown = 4  # detik
    
    # Variabel untuk mengatur AI tidur/bangun
    ai_active = False
    ai_start_time = 0
    ai_duration = 3 # Waktu AI melek/mendeteksi (dalam detik)

    while True:
        success, frame = cap.read()
        if not success:
            print("❌ Camera error")
            continue

        frame_to_show = frame # Default: Tampilkan kamera polos tanpa AI

        # ======================
        # 1. BACA SINYAL ESP32 (Tunggu aba-aba IR)
        # ======================
        if esp32 is not None and esp32.in_waiting:
            signal = esp32.readline().decode(errors='ignore').strip()
            
            if signal:
                print(f"📥 FROM ESP32: {signal}")

            if "DETECT" in signal:
                print("📡 IR DETECTED → BANGUNKAN AI SELAMA 3 DETIK!")
                ai_active = True
                ai_start_time = time.time() # Mulai hitung mundur

        # ======================
        # 2. JALANKAN AI HANYA JIKA SEDANG "DIBANGUNKAN"
        # ======================
        current_time = time.time()
        
        if ai_active:
            # Jika masih dalam durasi 3 detik sejak IR tersentuh
            if current_time - ai_start_time <= ai_duration:
                
                # AI MENDETEKSI & MENGGAMBAR KOTAK
                results = model.predict(source=frame, conf=0.8, show=False)
                frame_to_show = results[0].plot() 

                # Cek apakah sudah boleh ngirim data ke ESP32 (Cooldown)
                if current_time - last_sent_time > cooldown:
                    if len(results[0].boxes) > 0:
                        benda_terbaik = ""
                        keyakinan_tertinggi = 0.0

                        for box in results[0].boxes:
                            nama_benda = results[0].names[box.cls.item()].lower()
                            nilai_yakin = box.conf.item()

                            if (nama_benda in ["paper", "plastic"]) and nilai_yakin > keyakinan_tertinggi:
                                keyakinan_tertinggi = nilai_yakin
                                benda_terbaik = nama_benda

                        # KIRIM KE ESP32
                        if benda_terbaik == "paper":
                            print(f"▶️ SEND: PAPER ({keyakinan_tertinggi*100:.1f}%)")
                            esp32.write(b'1')
                            last_sent_time = current_time

                        elif benda_terbaik == "plastic":
                            print(f"▶️ SEND: PLASTIC ({keyakinan_tertinggi*100:.1f}%)")
                            esp32.write(b'2')
                            last_sent_time = current_time
                    else:
                        print("⚠️ Sedang mendeteksi, tapi belum ada benda yang jelas...")
            
            else:
                # Waktu habis, tidurkan AI kembali
                print("🛑 Waktu deteksi habis, AI kembali tidur.")
                ai_active = False

        # ======================
        # 3. STREAM KE WEB
        # ======================
        ret, buffer = cv2.imencode('.jpg', frame_to_show)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()

# ======================
# ROUTES
# ======================
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# ======================
# RUN APP
# ======================
if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)