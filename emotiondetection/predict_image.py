import cv2
from ultralytics import YOLO

# Load emotion model
model = YOLO("models/emotion_best.pt")

# Image path
image_path = "test.jpg"      # Replace with your image

# Predict
results = model.predict(
    source=image_path,
    imgsz=224,
    device=0,
    verbose=False
)

result = results[0]

emotion = result.names[result.probs.top1]
confidence = float(result.probs.top1conf)

print(f"Emotion : {emotion}")
print(f"Confidence : {confidence:.3f}")

img = cv2.imread(image_path)

cv2.putText(
    img,
    f"{emotion} ({confidence:.2f})",
    (20,40),
    cv2.FONT_HERSHEY_SIMPLEX,
    1,
    (0,255,0),
    2
)

cv2.imshow("Prediction", img)
cv2.waitKey(0)
cv2.destroyAllWindows()