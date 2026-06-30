from ultralytics import YOLO


def main():
    # Load the trained model
    model = YOLO(r"C:\Users\luhar\runs\classify\runs\emotion_cls-5\weights\best.pt")

    # Evaluate on the test set
    metrics = model.val(
        data="dataset",
        split="test",
        imgsz=224,
        batch=16,
        device=0
    )

    print("\nEvaluation Complete!")
    print(metrics)


if __name__ == "__main__":
    main()