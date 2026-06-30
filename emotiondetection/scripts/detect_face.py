import cv2
import os

# -----------------------------
# Paths
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(
    BASE_DIR,
    "..",
    "models",
    "face_detection_yunet_2023mar.onnx"
)

# -----------------------------
# Webcam
# -----------------------------
cap = cv2.VideoCapture(0)

ret, frame = cap.read()

if not ret:
    print("Unable to access webcam")
    exit()

h, w = frame.shape[:2]

# -----------------------------
# Load YuNet
# -----------------------------
detector = cv2.FaceDetectorYN.create(
    MODEL_PATH,
    "",
    (w, h),
    score_threshold=0.8,
    nms_threshold=0.3,
    top_k=5000
)

print("Press Q to quit")

while True:

    ret, frame = cap.read()

    if not ret:
        break

    h, w = frame.shape[:2]

    detector.setInputSize((w, h))

    _, faces = detector.detect(frame)

    if faces is not None:

        for face in faces:

            x, y, width, height = face[:4].astype(int)

            cv2.rectangle(
                frame,
                (x, y),
                (x + width, y + height),
                (0, 255, 0),
                2
            )

    cv2.imshow("YuNet Face Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()