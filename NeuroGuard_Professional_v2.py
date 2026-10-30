"""
NeuroGuard AI — Driver Drowsiness Monitor
-----------------------------------------
Run from the folder containing:
  - NeuroGuard_Professional.py (this file)
  - eye_model.keras
  - mouth_model.keras

Controls:
  Q / ESC : quit
  D       : toggle eye/mouth debug probabilities
  M       : toggle mirror mode

Note: This is an educational prototype, not a certified driving-safety device.
"""

from pathlib import Path
from collections import deque
import time
import threading
import math

import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf

# Sound alerts are available on Windows; the app still works without winsound.
try:
    import winsound
except ImportError:
    winsound = None


# ----------------------------- Configuration -----------------------------
APP_DIR = Path(__file__).resolve().parent
EYE_MODEL_PATH = APP_DIR / "eye_model.keras"
MOUTH_MODEL_PATH = APP_DIR / "mouth_model.keras"

IMG_SIZE = (96, 96)
CAMERA_INDEX = 0
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720

# Training notebook class order:
# Eye model:   Closed=0, Open=1
# Mouth model: no_yawn=0, yawn=1
EYE_CLOSED_CLASS = 0
EYE_OPEN_CLASS = 1
MOUTH_YAWN_CLASS = 1

CLOSED_THRESHOLD = 0.68  # CNN score shown for diagnostics only
EAR_CLOSED_THRESHOLD = 0.19  # Landmark eye-openness threshold; may need calibration
YAWN_THRESHOLD = 0.72
CLOSED_DURATION_SECONDS = 1.35
HISTORY_SIZE = 5
ALARM_REPEAT_SECONDS = 1.8

# BGR colors
BG = (22, 25, 35)
WHITE = (245, 247, 250)
MUTED = (165, 174, 190)
GREEN = (75, 205, 135)
ORANGE = (30, 165, 245)
RED = (65, 75, 245)
BLUE = (235, 160, 70)


# ------------------------------- Utilities -------------------------------
def load_model_checked(path: Path):
    if not path.exists():
        raise FileNotFoundError(
            f"Model file not found: {path}\n"
            "Put this Python file in the same folder as the .keras model files."
        )
    try:
        model = tf.keras.models.load_model(str(path), compile=False)
    except Exception as exc:
        raise RuntimeError(f"Could not load model '{path.name}': {exc}") from exc

    output_shape = model.output_shape
    if isinstance(output_shape, list):
        raise ValueError(f"Unexpected multiple outputs in {path.name}: {output_shape}")
    if len(output_shape) < 2 or output_shape[-1] != 2:
        raise ValueError(
            f"{path.name} should return 2 class probabilities, got output shape {output_shape}."
        )
    return model


def safe_crop(frame, landmarks, indices, padding_x=0.85, padding_y=0.95):
    """Crop a landmark region with generous padding and safe image boundaries."""
    height, width = frame.shape[:2]
    pts = [landmarks[i] for i in indices]
    xs = np.asarray([p.x * width for p in pts], dtype=np.float32)
    ys = np.asarray([p.y * height for p in pts], dtype=np.float32)

    x_min, x_max = float(xs.min()), float(xs.max())
    y_min, y_max = float(ys.min()), float(ys.max())
    box_w = max(2.0, x_max - x_min)
    box_h = max(2.0, y_max - y_min)

    pad_x = max(4, int(box_w * padding_x))
    pad_y = max(4, int(box_h * padding_y))

    x1 = max(0, int(x_min) - pad_x)
    y1 = max(0, int(y_min) - pad_y)
    x2 = min(width, int(np.ceil(x_max)) + pad_x)
    y2 = min(height, int(np.ceil(y_max)) + pad_y)

    if x2 - x1 < 4 or y2 - y1 < 4:
        return None
    crop = frame[y1:y2, x1:x2]
    return crop if crop.size else None


def euclidean_distance(point_a, point_b, width, height):
    ax, ay = point_a.x * width, point_a.y * height
    bx, by = point_b.x * width, point_b.y * height
    return math.hypot(ax - bx, ay - by)


def eye_aspect_ratio(landmarks, indices, width, height):
    # Six-point EAR: vertical eye opening divided by horizontal eye width.
    p1, p2, p3, p4, p5, p6 = [landmarks[i] for i in indices]
    horizontal = euclidean_distance(p1, p4, width, height)
    if horizontal < 1e-6:
        return 0.0
    vertical_1 = euclidean_distance(p2, p6, width, height)
    vertical_2 = euclidean_distance(p3, p5, width, height)
    return (vertical_1 + vertical_2) / (2.0 * horizontal)


def prepare_image(crop):
    """Keep RGB, 96x96 and uint8 0..255; the trained model rescales internally."""
    if crop is None or crop.size == 0:
        return None
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, IMG_SIZE, interpolation=cv2.INTER_AREA)
    return resized


def predict_probabilities(model, batch):
    """Return normalized two-class probabilities; reject invalid model output."""
    output = np.asarray(model(batch, training=False))
    if output.ndim != 2 or output.shape[1] != 2:
        raise ValueError(f"Expected model output shape (N, 2), got {output.shape}")
    output = output.astype(np.float32)
    if not np.all(np.isfinite(output)):
        raise ValueError("Model returned non-finite predictions.")
    # Softmax models should already sum to 1; normalize defensively.
    sums = output.sum(axis=1, keepdims=True)
    if np.any(sums <= 0):
        raise ValueError("Model returned invalid probability values.")
    return output / sums


def draw_text(img, text, x, y, scale=0.55, color=WHITE, thickness=1):
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                scale, color, thickness, cv2.LINE_AA)


def rounded_panel(frame, x1, y1, x2, y2, color=BG):
    # OpenCV-compatible filled panel with softened-looking edges.
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, -1, cv2.LINE_AA)


def draw_bar(frame, label, value, x, y, width, color):
    value = float(np.clip(value, 0.0, 1.0))
    draw_text(frame, label, x, y, 0.48, WHITE, 1)
    draw_text(frame, f"{value * 100:5.1f}%", x + width - 66, y, 0.48, WHITE, 1)
    bar_y = y + 10
    cv2.rectangle(frame, (x, bar_y), (x + width, bar_y + 9), (58, 64, 79), -1)
    fill = int(width * value)
    if fill > 0:
        cv2.rectangle(frame, (x, bar_y), (x + fill, bar_y + 9), color, -1)


def beep_worker(stop_event):
    if winsound is None:
        return
    while not stop_event.is_set():
        try:
            winsound.Beep(1350, 220)
        except Exception:
            return
        stop_event.wait(ALARM_REPEAT_SECONDS)


def start_alarm():
    global alarm_thread, alarm_stop_event
    if winsound is None or (alarm_thread is not None and alarm_thread.is_alive()):
        return
    alarm_stop_event = threading.Event()
    alarm_thread = threading.Thread(
        target=beep_worker, args=(alarm_stop_event,), daemon=True
    )
    alarm_thread.start()


def stop_alarm():
    global alarm_thread, alarm_stop_event
    if alarm_stop_event is not None:
        alarm_stop_event.set()
    alarm_thread = None
    alarm_stop_event = None


# ------------------------------ Main program ------------------------------
LEFT_EYE = [33, 133, 160, 159, 158, 157, 173, 144, 145, 153, 154, 155]
RIGHT_EYE = [362, 263, 387, 386, 385, 384, 398, 373, 374, 380, 381, 382]
LEFT_EAR_POINTS = [33, 160, 158, 133, 153, 144]
RIGHT_EAR_POINTS = [362, 385, 387, 263, 373, 380]
MOUTH = [61, 291, 0, 17, 13, 14, 78, 308]

eye_model = load_model_checked(EYE_MODEL_PATH)
mouth_model = load_model_checked(MOUTH_MODEL_PATH)

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.55,
    min_tracking_confidence=0.55,
)

# Try Windows' DirectShow camera backend first, then the default backend.
cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW) if hasattr(cv2, "CAP_DSHOW") else cv2.VideoCapture(CAMERA_INDEX)
if not cap.isOpened():
    cap.release()
    cap = cv2.VideoCapture(CAMERA_INDEX)
if not cap.isOpened():
    face_mesh.close()
    raise RuntimeError(
        "Webcam could not be opened. Close other camera apps and check Windows camera permissions."
    )

cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

eye_history = deque(maxlen=HISTORY_SIZE)
yawn_history = deque(maxlen=HISTORY_SIZE)
closed_since = None
last_alarm_time = 0.0
alarm_thread = None
alarm_stop_event = None
show_debug = True
mirror_mode = True
fps = 0.0
fps_last_time = time.perf_counter()
fps_frames = 0

print("=" * 58)
print(" NeuroGuard AI | Driver Drowsiness Monitor")
print("=" * 58)
print(f"Eye model:   {EYE_MODEL_PATH.name} | classes: Closed=0, Open=1")
print(f"Mouth model: {MOUTH_MODEL_PATH.name} | classes: no_yawn=0, yawn=1")
print("Controls: Q/ESC quit | D toggle probabilities | M mirror")
print(f"Eye alert uses MediaPipe EAR geometry (threshold={EAR_CLOSED_THRESHOLD:.2f}); CNN eye score is diagnostic.")
if winsound is None:
    print("Note: Windows sound alerts unavailable; visual alerts remain enabled.")
print("Starting camera...")

try:
    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            print("Camera frame could not be read. Check webcam connection.")
            break

        if mirror_mode:
            frame = cv2.flip(frame, 1)

        frame = cv2.resize(frame, (960, 540), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = face_mesh.process(rgb)
        rgb.flags.writeable = True

        status = "NORMAL"
        status_color = GREEN
        closed_prob = 0.0
        open_prob = 0.0
        yawn_prob = 0.0
        avg_closed = 0.0
        avg_yawn = 0.0
        closed_duration = 0.0
        left_ear = right_ear = avg_ear = 0.0
        eye_closed_geom = False
        face_found = bool(results.multi_face_landmarks)

        if face_found:
            landmarks = results.multi_face_landmarks[0].landmark
            h, w = frame.shape[:2]
            left_ear = eye_aspect_ratio(landmarks, LEFT_EAR_POINTS, w, h)
            right_ear = eye_aspect_ratio(landmarks, RIGHT_EAR_POINTS, w, h)
            avg_ear = (left_ear + right_ear) / 2.0
            # Use landmark geometry for alerts; CNN scores remain visible for debugging.
            eye_closed_geom = left_ear < EAR_CLOSED_THRESHOLD and right_ear < EAR_CLOSED_THRESHOLD
            left_crop = safe_crop(frame, landmarks, LEFT_EYE)
            right_crop = safe_crop(frame, landmarks, RIGHT_EYE)
            mouth_crop = safe_crop(frame, landmarks, MOUTH, padding_x=0.8, padding_y=1.3)

            eye_batch_items = []
            for crop in (left_crop, right_crop):
                prepared = prepare_image(crop)
                if prepared is not None:
                    eye_batch_items.append(prepared)

            if eye_batch_items:
                eye_batch = np.asarray(eye_batch_items, dtype=np.uint8)
                eye_probs = predict_probabilities(eye_model, eye_batch)
                closed_prob = float(np.mean(eye_probs[:, EYE_CLOSED_CLASS]))
                open_prob = float(np.mean(eye_probs[:, EYE_OPEN_CLASS]))
                eye_history.append(closed_prob)
            else:
                eye_history.clear()

            prepared_mouth = prepare_image(mouth_crop)
            if prepared_mouth is not None:
                mouth_batch = np.expand_dims(prepared_mouth, axis=0).astype(np.uint8)
                mouth_probs = predict_probabilities(mouth_model, mouth_batch)[0]
                yawn_prob = float(mouth_probs[MOUTH_YAWN_CLASS])
                yawn_history.append(yawn_prob)
            else:
                yawn_history.clear()

            avg_closed = float(np.mean(eye_history)) if eye_history else 0.0
            avg_yawn = float(np.mean(yawn_history)) if yawn_history else 0.0

            if eye_closed_geom:
                if closed_since is None:
                    closed_since = time.monotonic()
                closed_duration = time.monotonic() - closed_since
                if closed_duration >= CLOSED_DURATION_SECONDS:
                    status = "DROWSINESS ALERT"
                    status_color = RED
            else:
                closed_since = None
                closed_duration = 0.0
                stop_alarm()

            if avg_yawn >= YAWN_THRESHOLD and status != "DROWSINESS ALERT":
                status = "YAWNING DETECTED"
                status_color = ORANGE

            if status == "DROWSINESS ALERT":
                if time.monotonic() - last_alarm_time >= ALARM_REPEAT_SECONDS:
                    start_alarm()
                    last_alarm_time = time.monotonic()
            else:
                stop_alarm()

            # Face/eye outlines from landmark coordinates.
            h, w = frame.shape[:2]
            for idxs in (LEFT_EYE, RIGHT_EYE):
                pts = np.array(
                    [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in idxs],
                    dtype=np.int32
                )
                cv2.polylines(frame, [pts], True, status_color, 1, cv2.LINE_AA)

            if mouth_crop is not None:
                # No crop rectangle is shown because the crop is calculated internally.
                pass
        else:
            eye_history.clear()
            yawn_history.clear()
            closed_since = None
            closed_duration = 0.0
            stop_alarm()
            status = "FACE NOT DETECTED"
            status_color = ORANGE

        # FPS estimate
        fps_frames += 1
        now = time.perf_counter()
        elapsed = now - fps_last_time
        if elapsed >= 0.5:
            fps = fps_frames / elapsed
            fps_frames = 0
            fps_last_time = now

        # Dashboard overlay
        overlay = frame.copy()
        rounded_panel(overlay, 16, 16, 944, 126, (25, 29, 41))
        rounded_panel(overlay, 16, 140, 944, 260, (25, 29, 41))
        cv2.addWeighted(overlay, 0.90, frame, 0.10, 0, frame)

        draw_text(frame, "NEUROGUARD", 34, 48, 0.82, WHITE, 2)
        draw_text(frame, "AI  /  DRIVER MONITOR", 36, 73, 0.43, MUTED, 1)
        draw_text(frame, f"{fps:4.1f} FPS", 830, 46, 0.52, MUTED, 1)

        # Status pill
        pill_text = status
        cv2.rectangle(frame, (34, 87), (340, 116), status_color, -1, cv2.LINE_AA)
        draw_text(frame, pill_text, 46, 107, 0.55, (255, 255, 255), 2)

        if face_found:
            draw_bar(frame, "EYE CLOSURE (EAR ESTIMATE)", float(np.clip(1.0 - avg_ear / 0.30, 0.0, 1.0)), 370, 52, 250, RED if eye_closed_geom else GREEN)
            draw_bar(frame, "YAWN PROBABILITY", avg_yawn, 650, 52, 250, ORANGE if avg_yawn >= YAWN_THRESHOLD else BLUE)
        else:
            draw_text(frame, "Move into camera view", 370, 70, 0.56, ORANGE, 1)
            draw_text(frame, "Face landmarks are required", 370, 97, 0.44, MUTED, 1)

        draw_text(frame, "LIVE ANALYSIS", 34, 169, 0.46, MUTED, 1)
        draw_text(frame, f"Eyes: {'CLOSED' if eye_closed_geom else 'OPEN / NORMAL'}", 34, 198, 0.55, WHITE, 1)
        draw_text(frame, f"Closed duration: {closed_duration:.1f}s / {CLOSED_DURATION_SECONDS:.1f}s", 34, 226, 0.49, WHITE, 1)
        draw_text(frame, f"Alert threshold: {CLOSED_THRESHOLD:.2f}", 34, 248, 0.43, MUTED, 1)

        if show_debug:
            draw_text(frame, f"CNN closed: {closed_prob:.3f}   CNN open: {open_prob:.3f}", 520, 198, 0.48, WHITE, 1)
            draw_text(frame, f"Yawn raw: {yawn_prob:.3f}   Avg EAR: {avg_ear:.2f}", 520, 226, 0.48, WHITE, 1)
            draw_text(frame, "D: hide values", 520, 248, 0.43, MUTED, 1)
        else:
            draw_text(frame, "D: show probabilities", 520, 248, 0.43, MUTED, 1)

        draw_text(frame, "Q / ESC  Quit     D  Debug     M  Mirror", 30, 520, 0.45, MUTED, 1)
        cv2.imshow("NeuroGuard AI | Driver Drowsiness Monitor", frame)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        elif key == ord("d"):
            show_debug = not show_debug
        elif key == ord("m"):
            mirror_mode = not mirror_mode

finally:
    stop_alarm()
    cap.release()
    face_mesh.close()
    cv2.destroyAllWindows()
    print("NeuroGuard AI stopped.")
