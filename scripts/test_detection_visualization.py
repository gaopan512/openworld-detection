from pathlib import Path
import sys
import cv2
#获取项目根目录
PROJECT_ROOT = Path(__file__).resolve().parent.parent
#将项目根目录加入模块搜索路径
sys.path.insert(0,str(PROJECT_ROOT))
#导入检测和可视化模块
from src.detector import OpenWorldDetector
from src.visualizer import draw_detections

#设置模型和图片路径
model_path = PROJECT_ROOT / "models/yolov8s-worldv2.pt"
image_path = PROJECT_ROOT / "data/demo/bus.jpg"

#创建检测器
detector = OpenWorldDetector(
    model_path=model_path,
    device="mps"
)

#执行目标检测
detections = detector.detect(
    image_path=image_path,
    classes=["person","car","bus","bicycle","dog"],
    confidence=0.25
)
print("实际推理设备：", detector.model.predictor.device)

#输出检测结果数量
print(f"检测到{len(detections)}个目标")

#绘制检测结果
image = draw_detections(image_path,detections)

#保存可视化图片
output_path = PROJECT_ROOT / "outputs/detections/integration_test.jpg"
output_path.parent.mkdir(parents=True, exist_ok=True)
success = cv2.imwrite(str(output_path),image)
if not success:
    raise RuntimeError(f"图片保存失败：{output_path}")
print(f"可视化图片已保存：{output_path}")
# 输出每个目标的类别和置信度
for detection in detections:
    print(
        f"类别：{detection['class_name']}, "
        f"置信度：{detection['confidence']:.2f}"
    )