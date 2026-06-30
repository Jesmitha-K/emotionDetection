import cv2
import os
from ultralytics import YOLO

# -------------------------------------------------
# Paths
# -------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

FACE_MODEL = os.path.join(
    BASE_DIR,
    "models",
    "face_detection_yunet_2023mar.onnx"
)

EMOTION_MODEL = os.path.join(
    BASE_DIR,
    "models",
    "emotion_best.pt"
)

# -------------------------------------------------
# Load Emotion Model
# -------------------------------------------------

emotion_model = YOLO(EMOTION_MODEL)

# -------------------------------------------------
# Webcam
# -------------------------------------------------

cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)

ret, frame = cap.read()

if not ret:
    print("Cannot open webcam")
    exit()

h, w = frame.shape[:2]

# -------------------------------------------------
# YuNet Face Detector
# -------------------------------------------------

face_detector = cv2.FaceDetectorYN.create(
    FACE_MODEL,
    "",
    (w, h),
    score_threshold=0.8,
    nms_threshold=0.3,
    top_k=5000
)

print("Press Q to Quit")

while True:

    ret, frame = cap.read()

    if not ret:
        break

    h, w = frame.shape[:2]

    face_detector.setInputSize((w, h))

    _, faces = face_detector.detect(frame)

    if faces is not None:

        for face in faces:

            x, y, fw, fh = face[:4].astype(int)

            # -------------------------
            # Crop Face
            # -------------------------

            face_crop = frame[y:y+fh, x:x+fw]

            if face_crop.size == 0:
                continue

            # -------------------------
            # Emotion Prediction
            # -------------------------

            results = emotion_model.predict(
                source=face_crop,
                imgsz=224,
                verbose=False
            )

            result = results[0]

            emotion = result.names[result.probs.top1]
            confidence = float(result.probs.top1conf)

            # -------------------------
            # Draw Bounding Box
            # -------------------------

            cv2.rectangle(
                frame,
                (x, y),
                (x+fw, y+fh),
                (0,255,0),
                2
            )

            label = f"{emotion} ({confidence:.2f})"

            cv2.putText(
                frame,
                label,
                (x, y-10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0,255,0),
                2
            )

    cv2.imshow("Real-Time Emotion Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()