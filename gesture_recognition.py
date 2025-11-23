import cv2
import mediapipe as mp
import time

# ==============================================================================
# INSTRUKSI INSTALASI LIBRARY
# ==============================================================================
# Sebelum menjalankan program ini, pastikan Anda telah menginstal library yang dibutuhkan.
# Buka terminal atau command prompt dan jalankan perintah berikut:
#
# pip install opencv-python mediapipe
#
# ==============================================================================

class GestureRecognitionApp:
    def __init__(self):
        """
        Inisialisasi konfigurasi MediaPipe dan variabel gesture.
        """
        # Inisialisasi MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # Inisialisasi MediaPipe Face Detection
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection = self.mp_face_detection.FaceDetection(
            min_detection_confidence=0.5
        )

        # Utilitas menggambar (untuk bounding box dan landmarks)
        self.mp_draw = mp.solutions.drawing_utils

        # ==========================================================================
        # KONFIGURASI GESTURE (MAPPING)
        # ==========================================================================
        # Format Key: Tuple (Jempol, Telunjuk, Tengah, Manis, Kelingking)
        # 1 = Terangkat/Terbuka, 0 = Tertutup/Ditekuk
        # Anda bisa mengubah atau menambahkan aturan gesture di sini.
        self.gesture_map = {
            (1, 1, 1, 1, 1): "Kepalan Tangan (Fist)",
            (1, 0, 0, 0, 0): "Mantap",
            (0, 1, 0, 0, 0): "Nama Saya",
            (0, 1, 1, 0, 0): "Ersaf Sirazi Arifin",
            (0, 1, 1, 1, 0): "Kelas",
            (0, 1, 1, 1, 1): "XI PPLG 2",
            (1, 1, 1, 1, 1): "Halo",
            (0, 1, 0, 0, 1): "Rock n Roll",
            (1, 1, 0, 0, 1): "XI PPLG 2",
            (0, 0, 1, 0, 0): "NGENTOT LO ANJING TAI BANGSAT",
        }

    def detect_fingers(self, hand_landmarks, handedness_label):
        """
        Mendeteksi status setiap jari (terbuka/tertutup).
        
        Args:
            hand_landmarks: Objek landmarks dari MediaPipe.
            handedness_label: Label tangan ('Left' atau 'Right').
            
        Returns:
            List[int]: Status 5 jari [Jempol, Telunjuk, Tengah, Manis, Kelingking]
                       1 jika terbuka, 0 jika tertutup.
        """
        fingers = []
        
        # ID Landmarks ujung jari (Tip)
        # Jempol: 4, Telunjuk: 8, Tengah: 12, Manis: 16, Kelingking: 20
        tip_ids = [4, 8, 12, 16, 20]

        # --- Logika Jempol (Thumb) ---
        # Jempol bergerak menyamping, bukan ke atas/bawah seperti jari lain.
        # Kita membandingkan posisi x ujung jempol (4) dengan pangkal jempol (3 atau 2).
        # Perlu memperhatikan tangan kiri vs kanan.
        
        # Catatan: MediaPipe menganggap 'Left' adalah tangan kiri subjek.
        # Namun pada mode selfie (kamera depan), gambar seringkali di-mirror.
        # Logika di bawah mengasumsikan gambar tidak di-flip kembali secara manual (standar webcam).
        
        if handedness_label == 'Right':
            # Untuk tangan kanan, jika ujung jempol lebih ke kiri (x lebih kecil) dari sendi, maka terbuka
            if hand_landmarks.landmark[tip_ids[0]].x < hand_landmarks.landmark[tip_ids[0] - 1].x:
                fingers.append(1)
            else:
                fingers.append(0)
        else:
            # Untuk tangan kiri, jika ujung jempol lebih ke kanan (x lebih besar) dari sendi, maka terbuka
            if hand_landmarks.landmark[tip_ids[0]].x > hand_landmarks.landmark[tip_ids[0] - 1].x:
                fingers.append(1)
            else:
                fingers.append(0)

        # --- Logika 4 Jari Lainnya (Telunjuk s/d Kelingking) ---
        # Jari dianggap terbuka jika posisi y ujung jari (tip) lebih tinggi (nilai y lebih kecil)
        # daripada posisi y sendi tengah (pip - landmark id dikurangi 2).
        # Koordinat Y pada gambar: 0 di atas, 1 di bawah. Jadi y_tip < y_pip berarti jari naik.
        
        for id in range(1, 5):
            if hand_landmarks.landmark[tip_ids[id]].y < hand_landmarks.landmark[tip_ids[id] - 2].y:
                fingers.append(1)
            else:
                fingers.append(0)

        return fingers

    def detect_gesture(self, fingers):
        """
        Mencocokkan kombinasi jari dengan dictionary gesture.
        
        Args:
            fingers (List[int]): List status jari, contoh [1, 0, 0, 0, 0]
            
        Returns:
            str: Nama gesture atau "Unknown Gesture" jika tidak dikenali.
        """
        # Ubah list ke tuple agar bisa dijadikan key dictionary
        finger_tuple = tuple(fingers)
        
        # Ambil dari dictionary, default ke "Unknown" jika tidak ada
        return self.gesture_map.get(finger_tuple, "Unknown Gesture")

    def run(self):
        """
        Menjalankan loop utama program: membaca webcam, mendeteksi, dan menampilkan output.
        """
        cap = cv2.VideoCapture(0) # 0 biasanya adalah ID webcam default
        
        if not cap.isOpened():
            print("Error: Tidak dapat mengakses webcam.")
            return

        print("Program berjalan... Tekan 'q' untuk keluar.")

        while True:
            success, img = cap.read()
            if not success:
                print("Gagal membaca frame dari webcam.")
                break

            # Flip gambar secara horizontal agar seperti cermin (opsional, tapi lebih natural)
            img = cv2.flip(img, 1)
            
            # Konversi BGR ke RGB karena MediaPipe membutuhkan input RGB
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            h, w, c = img.shape

            # ==================================================================
            # 1. DETEKSI WAJAH (FACE DETECTION)
            # ==================================================================
            face_results = self.face_detection.process(img_rgb)
            
            if face_results.detections:
                for detection in face_results.detections:
                    # Menggambar bounding box wajah
                    # MediaPipe mengembalikan koordinat relatif (0.0 - 1.0), perlu dikali dimensi gambar
                    bboxC = detection.location_data.relative_bounding_box
                    ih, iw, _ = img.shape
                    x, y, w_box, h_box = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                                         int(bboxC.width * iw), int(bboxC.height * ih)
                    
                    # Gambar kotak di sekitar wajah
                    cv2.rectangle(img, (x, y), (x + w_box, y + h_box), (255, 0, 255), 2)
                    cv2.putText(img, 'Face', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)

            # ==================================================================
            # 2. DETEKSI TANGAN (HAND DETECTION)
            # ==================================================================
            hand_results = self.hands.process(img_rgb)

            if hand_results.multi_hand_landmarks:
                # Loop untuk setiap tangan yang terdeteksi
                for hand_idx, hand_landmarks in enumerate(hand_results.multi_hand_landmarks):
                    
                    # Dapatkan label tangan (Left/Right)
                    # Note: index classification sesuai urutan deteksi
                    handedness_label = "Right" # Default
                    if hand_results.multi_handedness:
                        # Mengambil label dari hasil klasifikasi
                        handedness = hand_results.multi_handedness[hand_idx].classification[0].label
                        handedness_label = handedness

                    # Gambar landmarks tangan di layar
                    self.mp_draw.draw_landmarks(img, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

                    # Hitung status jari [1, 0, 1, ...]
                    fingers_status = self.detect_fingers(hand_landmarks, handedness_label)
                    
                    # Dapatkan nama gesture berdasarkan status jari
                    gesture_text = self.detect_gesture(fingers_status)

                    # ==========================================================
                    # TAMPILKAN OUTPUT TEKS
                    # ==========================================================
                    # Koordinat untuk menampilkan teks (di dekat tangan atau pojok layar)
                    # Kita ambil posisi pergelangan tangan (landmark 0) untuk posisi teks
                    wrist = hand_landmarks.landmark[0]
                    cx, cy = int(wrist.x * w), int(wrist.y * h)

                    # Tampilkan Status Jari (Debug info)
                    status_str = str(fingers_status)
                    cv2.putText(img, status_str, (cx, cy + 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

                    # Tampilkan Nama Gesture (Hasil Utama) dengan Text Wrapping
                    max_width = 200  # Lebar maksimum teks sebelum turun baris
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 0.8
                    thickness = 2
                    
                    words = gesture_text.split(' ')
                    lines = []
                    current_line = ""
                    
                    for word in words:
                        test_line = current_line + word + " "
                        (text_w, text_h), _ = cv2.getTextSize(test_line, font, font_scale, thickness)
                        if text_w > max_width and current_line != "":
                            lines.append(current_line)
                            current_line = word + " "
                        else:
                            current_line = test_line
                    lines.append(current_line)

                    # Hitung tinggi kotak background
                    line_height = 30
                    box_height = len(lines) * line_height
                    
                    # Koordinat kotak (tumbuh ke atas dari posisi awal)
                    box_x1 = cx - 20
                    box_y2 = cy - 10
                    box_y1 = box_y2 - box_height - 10 # -10 untuk padding atas
                    box_x2 = box_x1 + max_width + 30

                    # Gambar background hitam
                    cv2.rectangle(img, (box_x1, box_y1), (box_x2, box_y2), (0, 0, 0), cv2.FILLED)

                    # Gambar teks per baris
                    for i, line in enumerate(lines):
                        y_pos = box_y1 + 30 + (i * line_height)
                        cv2.putText(img, line.strip(), (box_x1 + 10, y_pos - 5), 
                                    font, font_scale, (0, 255, 0), thickness)

            # Tampilkan frame akhir
            cv2.imshow("Gesture & Face Recognition System", img)

            # Tekan 'q' untuk keluar
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    app = GestureRecognitionApp()
    app.run()
