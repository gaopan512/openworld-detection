from pathlib import Path
import argparse
import sys

# 获取项目根目录
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# 将项目根目录加入模块搜索路径
sys.path.insert(0, str(PROJECT_ROOT))

from src.class_parser import parse_classes
from src.detector import OpenWorldDetector
from src.visualizer import draw_detections
from src.image_io import write_image


def main():
    parser = argparse.ArgumentParser(description="交互式检测与绘图验证")
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    args = parser.parse_args()
    # 接收用户输入，并解析类别
    text = input(
        "请输入英文检测类别，用逗号分隔"
        "（例如 person, bus, traffic light）：\n"
    )

    try:
        classes = parse_classes(text)
    except (TypeError, ValueError) as error:
        print(f"输入错误：{error}")
        return

    print(f"本次检测类别：{classes}")

    # 设置模型和图片路径
    model_path = PROJECT_ROOT / "models/yolov8s-worldv2.pt"
    image_path = PROJECT_ROOT / "data/demo/bus.jpg"

    # 默认使用 CPU；可显式指定 MPS 或 CUDA。
    detector = OpenWorldDetector(
        model_path=model_path,
        device=args.device
    )

    # 使用解析后的类别执行检测
    detections = detector.detect(
        image_path=image_path,
        classes=classes,
        confidence=0.25
    )

    print("实际推理设备：", detector.model.predictor.device)
    print(f"检测到 {len(detections)} 个目标")

    for detection in detections:
        print(
            f"类别：{detection['class_name']}, "
            f"置信度：{detection['confidence']:.2f}"
        )

    if not detections:
        print("当前提示词和置信度阈值下未检测到目标。")

    # 绘制检测结果
    image = draw_detections(image_path, detections)

    # 保存图片；再次运行会覆盖同名结果
    output_path = (
        PROJECT_ROOT / "outputs/detections/integration_test.jpg"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    write_image(output_path, image)

    print(f"可视化图片已保存：{output_path}")


if __name__ == "__main__":
    main()
