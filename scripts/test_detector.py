from pathlib import Path
import argparse
import  sys
#获取项目根目录
PROJECT_ROOT=Path(__file__).resolve().parent.parent

#将项目根目录加入Python模块搜索路径
sys.path.insert(0,str(PROJECT_ROOT))

#导入我们自己编写的检测类
from src.detector import OpenWorldDetector


def main():
    parser = argparse.ArgumentParser(description="验证真实检测器")
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    args = parser.parse_args()
    #初始化检测器
    detector = OpenWorldDetector(
        model_path=PROJECT_ROOT / "models/yolov8s-worldv2.pt",
        device=args.device
    )

    #执行目标检测
    detections=detector.detect(
        image_path=PROJECT_ROOT/"data/demo/bus.jpg",
        classes=["person","car","bicycle","dog"],
        confidence=0.25
    )
    #输出检测结果
    print(f"\n检测到{len(detections)}个目标：")
    for i,detection in enumerate(detections,start=1):
        print(
            f"{i}.类别：{detection['class_name']},"
            f"置信度：{detection['confidence']:.2f},"
            f"坐标：{detection['bbox']}"
        )
    print("\n检测模块测试完成！")
if __name__=="__main__":
    main()
