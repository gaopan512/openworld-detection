"""真实模型实验：单独改变提示词或阈值，保存图片和原始检测记录。"""

import argparse
import hashlib
import json
import platform
import sys
from datetime import datetime
from importlib.metadata import version
from pathlib import Path
from zoneinfo import ZoneInfo


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    parser.add_argument("--image", type=Path,
                        default=PROJECT_ROOT / "data/demo/bus.jpg")
    args = parser.parse_args()

    import torch

    from src.class_parser import parse_classes
    from src.detector import OpenWorldDetector
    from src.visualizer import draw_detections
    from src.image_io import write_image

    image_path = args.image.resolve()
    model_path = PROJECT_ROOT / "models/yolov8s-worldv2.pt"
    if not image_path.is_file():
        parser.error(f"图片不存在：{image_path}")
    if not model_path.is_file():
        parser.error(f"模型不存在：{model_path}")
    if args.device == "mps" and not torch.backends.mps.is_available():
        parser.error("当前进程中 MPS 不可用，请使用 --device cpu")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("当前进程中 CUDA 不可用，请使用 --device cpu")

    started = datetime.now(ZoneInfo("Asia/Shanghai"))
    run_id = started.strftime("%Y%m%d_%H%M%S_%f")
    record_dir = PROJECT_ROOT / "outputs/experiments" / run_id
    image_dir = PROJECT_ROOT / "outputs/detections/experiments" / run_id
    record_dir.mkdir(parents=True, exist_ok=False)
    image_dir.mkdir(parents=True, exist_ok=False)
    record_path = record_dir / "results.json"

    # 前三组只改变阈值，后两组分别只替换一个类别短语。
    # 这些是实验条件，不是系统允许输入的类别白名单。
    cases = [
        ("baseline", "person, bus", 0.25),
        ("threshold_050", "person, bus", 0.50),
        ("threshold_075", "person, bus", 0.75),
        ("prompt_pedestrian", "pedestrian, bus", 0.25),
        ("prompt_coach", "person, coach", 0.25),
    ]

    record = {
        "run_id": run_id,
        "started_at": started.isoformat(),
        "status": "running",
        "image": str(image_path),
        "image_sha256": file_sha256(image_path),
        "model": str(model_path),
        "model_sha256": file_sha256(model_path),
        "requested_device": args.device,
        "python": platform.python_version(),
        "packages": {name: version(name) for name in
                     ("ultralytics", "torch", "opencv-python")},
        "notes": "单张图片的定性对比；没有人工真值标注，不计算准确率或 mAP；不作性能基准。",
        "cases": [],
    }

    def write_record():
        temporary_path = record_path.with_suffix(".json.tmp")
        temporary_path.write_text(
            json.dumps(record, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8"
        )
        temporary_path.replace(record_path)

    write_record()
    try:
        detector = OpenWorldDetector(model_path, device=args.device)
        for name, prompt, threshold in cases:
            classes = parse_classes(prompt)
            detections = detector.detect(image_path, classes, threshold)
            annotated = draw_detections(image_path, detections)
            output_path = image_dir / f"{name}.png"
            write_image(output_path, annotated)

            record["cases"].append({
                "name": name,
                "prompt": prompt,
                "classes": classes,
                "confidence_threshold": threshold,
                "actual_device": str(detector.model.predictor.device),
                "result_image": str(output_path.relative_to(PROJECT_ROOT)),
                "detections": detections,
            })
            write_record()
            print(f"已完成：{name}，结果图片：{output_path}", flush=True)
        record["status"] = "complete"
    except Exception as error:
        record["status"] = "failed"
        record["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        record["finished_at"] = datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()
        write_record()

    print(f"实验记录已保存：{record_path}")


if __name__ == "__main__":
    main()
