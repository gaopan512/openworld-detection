from pathlib import Path
import sys
import cv2

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.visualizer import draw_detections


# 测试图片
image_path = PROJECT_ROOT / "data/demo/bus.jpg"

# 测试案例
test_cases = {
    "empty": [],

    "long_label": [
        {
          "class_name": "a very long object category name " * 10,
            "confidence": 0.95,
            "bbox": [50, 100, 200, 300]
        }
    ],

    "top_edge": [
        {
            "class_name": "person",
            "confidence": 0.92,
            "bbox": [100, 0, 250, 200]
        }
    ],

    "right_edge": [
        {
            "class_name": "traffic light",
            "confidence": 0.87,
            "bbox": [750, 100, 800, 300]
        }
    ]
}

# 输出目录
output_dir = PROJECT_ROOT / "outputs/detections/edge_tests"
output_dir.mkdir(parents=True, exist_ok=True)

# 执行测试
for name, detections in test_cases.items():

    image = draw_detections(image_path, detections)

    output_path = output_dir / f"{name}.jpg"

    success = cv2.imwrite(str(output_path), image)

    if not success:
        raise RuntimeError(f"图片保存失败：{output_path}")

    print(f"{name}: 测试图片已保存")