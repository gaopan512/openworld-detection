from ultralytics import YOLOWorld
from pathlib import Path
import argparse
PROJECT_ROOT = Path(__file__).resolve().parent.parent

def main():
    parser = argparse.ArgumentParser(description="验证 YOLO-World 原生接口")
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    args = parser.parse_args()
    run(args.device)


def run(device):
    model = YOLOWorld(str(PROJECT_ROOT / "models/yolov8s-worldv2.pt"))
    model.set_classes(["person", "car", "bicycle", "dog"])
    results = model.predict(
        source=str(PROJECT_ROOT / "data/demo/bus.jpg"), conf=0.25,
        device=device, save=True,
        project=str(PROJECT_ROOT / "outputs/detections"),
        name="native_test", exist_ok=True,
    )
    for result in results:
        for box in result.boxes:
            print(f"类别：{result.names[int(box.cls[0])]}, 置信度：{float(box.conf[0]):.2f}")
    print("检测完成！")


if __name__ == "__main__":
    main()
