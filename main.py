import cv2
import mediapipe as mp

# --- Inisialisasi MediaPipe Hands ---
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(
    max_num_hands=2,  # deteksi sampai 2 tangan
    model_complexity=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

BLUR_KERNEL = (55, 55)  # makin besar (ganjil) makin blur
WINDOW_TITLE = "foto kita blurrr"


def is_peace_sign(landmarks):
    """Cek apakah landmark tangan membentuk peace sign (telunjuk & tengah lurus,
    manis & kelingking terlipat)."""
    lm = landmarks.landmark

    def extended(tip_idx, pip_idx):
        return lm[tip_idx].y < lm[pip_idx].y

    def folded(tip_idx, pip_idx):
        return lm[tip_idx].y > lm[pip_idx].y

    index_extended = extended(8, 6)
    middle_extended = extended(12, 10)
    ring_folded = folded(16, 14)
    pinky_folded = folded(20, 18)

    return index_extended and middle_extended and ring_folded and pinky_folded


def main():
    cap = cv2.VideoCapture(0)  # 0 = kamera default

    if not cap.isOpened():
        print("Gagal membuka kamera.")
        return

    while True:
        success, frame = cap.read()
        if not success:
            print("Gagal membaca frame dari kamera.")
            break

        frame = cv2.flip(frame, 1)  # efek cermin, lebih natural
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        peace_detected = False

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Gambar skeleton untuk setiap tangan yang terdeteksi
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_styles.get_default_hand_landmarks_style(),
                    mp_styles.get_default_hand_connections_style(),
                )
                if is_peace_sign(hand_landmarks):
                    peace_detected = True

        if peace_detected:
            frame = cv2.GaussianBlur(frame, BLUR_KERNEL, 0)

        cv2.imshow(WINDOW_TITLE, frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()