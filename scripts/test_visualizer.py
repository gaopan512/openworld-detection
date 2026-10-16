from pathlib import Path
import sys
import cv2

#获取项目根目录
PROJECT_ROOT = Path(__file__).resolve().parent.parent

#将项目根目录加入 python 模块搜索路径
sys.path.insert(0,str(PROJECT_ROOT))

from src.visualizer import draw_detections

#测试图片
image_path = PROJECT_ROOT / "data/demo/bus.jpg"

#模拟检测结果
detections = [
    {
        "class_name": "person",
        "confidence": 0.95,
        "bbox": [100, 50, 300, 400]
    }
]

#调用可视化函数
image = draw_detections(image_path,detections)

#保存测试图片
output_path = PROJECT_ROOT / "outputs/detections/visualizer_test.jpg"

success = cv2.imwrite(str(output_path),image)

if not success:
    raise RuntimeError(f"图片保存失败：{output_path}")
print(f"可视化结果已保存：{output_path}")