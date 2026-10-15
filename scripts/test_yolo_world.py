from ultralytics import YOLOWorld
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
# 加载yw v2预训练模型
model = YOLOWorld(
    str(PROJECT_ROOT / "models/yolov8s-worldv2.pt")
)
#设置需要检测的目标类别
model.set_classes(["person","car","bicycle","dog"])
#执行目标检测
results=model.predict(
    source=str(PROJECT_ROOT / "data/demo/bus.jpg"),
    conf=0.25,#置信度阈值
    device="mps",
    save=True,
    project=str(PROJECT_ROOT / "outputs" / "detections"),
    name="mps_test",
    exist_ok=True,
)
#输出检测结果
for result in results:
    print("\n检测结果：")
    for box in result.boxes:
        class_id=int(box.cls[0])
        confidence=float(box.conf[0])
        print(
            f"类别：{result.names[class_id]},"
            f"置信度：{confidence:.2f}"
        )
print("\n检测完成！")