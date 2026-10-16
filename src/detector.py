from pathlib import Path
from ultralytics import YOLOWorld


class OpenWorldDetector:
    """YOLO-World. 开放词汇目标检测器。"""
    def __init__(self,model_path,device="cpu"):
        #检查模型文件是否存在
        self.model_path=Path(model_path).resolve()

        if not self.model_path.is_file():
            raise FileNotFoundError(
                f"模型文件不存在：{self.model_path}"
            )
        #保存计算设备
        self.device = device
        #加载训练模型（只执行一次）
        self.model=YOLOWorld(str(self.model_path))
        print(f"模型加载成功，计算设备：{self.device}")
    def detect (self,image_path,classes,confidence=0.25):
        """检测图片并返回结构化结果。"""
        image_path=Path(image_path).resolve()

        #检查图片
        if not image_path.is_file():
            raise FileNotFoundError(
                f"图片不存在：{image_path}"
            )
        #检查检测类别
        if not isinstance(classes,(list,tuple)) or not classes:
            raise ValueError("检测类别必须是非空列表或元组")


        if not all(isinstance(c,str)and c.strip()for c in classes):
            raise ValueError("检测类别必须是非空字符串")

        #检查置信度阈值
        if not 0<=confidence<=1:
            raise ValueError("置信度必须在0到1之间")
        #设置检测类别
        self.model.set_classes(classes)

        #执行推理
        results=self.model.predict(
            source=str(image_path),
            conf=confidence,
            device=self.device,
            verbose=False,
        )
        #整理检测结果
        detections=[]

        for result in results:
            for box in result.boxes:
                class_id =int(box.cls[0])

                detections.append({
                    "class_name":result.names[class_id],
                    "confidence":float(box.conf[0]),
                    "bbox":box.xyxy[0].tolist(),
                })
        return detections