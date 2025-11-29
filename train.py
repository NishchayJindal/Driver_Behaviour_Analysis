from ultralytics import YOLO

def main():
    # Load YOLOv8 Small (best choice for speed & accuracy balance)
    model = YOLO('yolov8s.pt')

    model.train(
        data=r"C:\nish\gtsdb.yaml",   # <-- YOUR UPDATED PATH
        epochs=35,
        imgsz=640,
        batch=16,
        device=0,                        # Use GPU 0
        name='gtsdb_v8s'                 # Output folder name
    )

if __name__ == '__main__':
    main()
