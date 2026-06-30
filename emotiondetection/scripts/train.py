from ultralytics import YOLO


def main():
    # Load pretrained YOLO11 Classification model
    model = YOLO("yolo11n-cls.pt")

    # Train
    results = model.train(
        data="dataset",          # dataset folder
        epochs=50,               # train for 50 epochs
        imgsz=224,               # input image size
        batch=16,                # suitable for GTX 1650
        device=0,                # GPU
        workers=2,               # dataloader workers
        optimizer="AdamW",       # optimizer
        lr0=0.001,               # initial learning rate
        weight_decay=5e-4,
        patience=10,             # early stopping
        pretrained=True,
        verbose=True,
        plots=True,
        save=True,
        project="runs",
        name="emotion_cls"
    )

    print("\nTraining Completed Successfully!")
    print(results.save_dir)


if __name__ == "__main__":
    main()