# 基于 YOLO-World 的开放词汇目标检测与可视化分析系统

使用 YOLO-World v2 与 PyQt6，根据自由输入的英文类别短语检测单张图片中的目标，并展示、保存带检测框的结果。

课程项目 V1 于 2026-10-22 完成结项整理，后续进入维护阶段。功能实现、实验结果及验证边界见[结项说明](docs/项目结项说明.md)。

- [课程报告（Markdown）](docs/开放词汇目标检测系统课程报告.md)
- [课程报告（Word）](docs/开放词汇目标检测系统课程报告.docx)
- [实验统计汇总](docs/实验摘要/统计汇总.json) · [图片来源与许可](docs/实验摘要/图片来源与许可.csv)

![系统界面](docs/课程报告配图/gui.png)

## 功能

- 打开或拖入 JPG、JPEG、PNG、BMP 图片。
- 自由输入英文检测类别，支持多词短语，无固定类别白名单。
- 调整置信度阈值，后台执行检测。
- 显示检测框、类别、置信度、检测明细、耗时和设备信息。
- 原图与检测结果页签切换，支持滚轮缩放、拖动平移、双击适应窗口。
- 点击类别标签或明细表中的目标，高亮对应检测框。
- 可选输入停顿后自动检测，默认关闭。
- 保存完整分辨率的 PNG、JPEG 结果图片。

系统以单张图片为输入，通过英文类别短语和置信度阈值控制检测目标与结果筛选。

## 安装与启动

需要 Python 3.11、Git 和网络连接。使用 Conda：

```bash
git clone https://github.com/gaopan512/openworld-detection.git
cd openworld-detection
conda env create -f environment.yml
conda activate openworld-detection
python -B main.py
```

已有独立 Python 3.11 环境时，在项目根目录安装依赖并启动：

```bash
python -m pip install -r requirements.txt
python -B main.py
```

CUDA 推理需要兼容的 NVIDIA 显卡、驱动及 CUDA 版 PyTorch；请按目标设备选择适用的 PyTorch 安装组合。Linux 图形界面需要桌面显示服务及 Qt 系统依赖。

## 模型准备

首次使用默认模型检测时，程序在后台下载并校验 YOLO-World 与 CLIP 权重，状态栏显示下载进度。两份文件合计约 363 MiB，后续复用本地文件。

也可在项目根目录提前准备权重：

```bash
python -B -m src.model_assets
```

该命令会显示两份文件的保存位置：

| 权重 | 下载来源 | 保存位置 |
| --- | --- | --- |
| YOLOv8s-Worldv2 | [Ultralytics v8.3.0 Release](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8s-worldv2.pt) | 项目内 `models/yolov8s-worldv2.pt` |
| CLIP ViT-B/32 | [官方权重](https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt) | 当前 Ultralytics 缓存的 `WEIGHTS_DIR/clip/ViT-B-32.pt` |

默认权重采用固定版本，并检查文件大小和 SHA-256。下载失败最多尝试 3 次，成功校验后才替换目标文件。若获取失败，可检查网络后重新检测或重跑预下载命令，也可按错误提示中的官方地址与保存位置手动准备文件。

离线使用前，安装依赖并准备 YOLO 与 CLIP 两份权重。权重通过上述官方地址下载。

其他兼容的本地 `.pt` 模型可放入 `models/`，重启程序后在下拉框中选择；其权重与额外依赖需自行准备。

## 使用方法

1. 点击“打开图片”或拖入图片，可先使用 `data/demo/bus.jpg`。
2. 输入英文检测目标，例如 `person, bus, traffic light`。
3. 设置置信度阈值，默认 `0.25`。
4. 点击“开始检测”，等待结果显示。
5. 点击“保存结果”，选择位置、文件名和 PNG/JPEG 格式。

检测期间，界面锁定图片切换、参数修改与窗口关闭，任务结束后恢复。首次检测会准备权重并初始化模型，状态栏显示执行进度。

### 提示词

- 使用逗号分隔不同类别，兼容中文逗号。
- 保留多词短语，例如 `fire extinguisher`。
- 自动清理多余空白和空项，按忽略大小写去重。
- 空输入或包含中文汉字时会提示错误；不进行中文翻译。

### 预览与高亮

原图与结果可通过页签切换。滚轮缩放，拖动平移，双击恢复适应窗口。点击类别标签可高亮同类目标，再次点击取消；点击明细表中的一行可高亮对应目标。标签虚线表示当前阈值下该类别未检出。

更改输入后，旧结果仍可查看与保存，类别标签显示“待检测”。开始新检测时清空旧结果。高亮只影响预览，不写入保存图片。

开启“输入停顿后自动检测”后，修改有效输入或参数并停顿约 0.9 秒，会执行新一轮检测。打开图片或保存结果的对话框期间暂停自动检测。

### 计算设备

默认自动按 MPS、CUDA、CPU 顺序选择可用设备，也可手动选择。自动选择加速设备后若加载或推理失败，会显示原因并用 CPU 重试一次；手动选择设备时直接报告错误。下载失败不会触发设备回退。

结果显示最终使用的设备。检测耗时包含权重准备、模型加载、文本编码、推理及可能发生的 CPU 重试，不包含绘图和保存。

### 保存结果

默认保存目录为 `outputs/detections/`，默认文件名为 `原图名_detected.png`。可自行选择其他位置或 JPEG 格式。保存使用完整分辨率结果，程序阻止覆盖当前原图，覆盖其他已有文件前会确认。

未检测到目标时也可保存对应结果图片。检测失败时会显示错误信息，处理后可再次检测。

## 项目结构

```text
main.py                      # 启动入口
src/class_parser.py          # 英文提示词解析
src/detector.py              # 模型加载与目标检测
src/device.py                # 计算设备选择
src/model_assets.py          # 默认权重下载与校验
src/image_io.py              # 图片读写
src/visualizer.py            # 检测框与标签绘制
src/gui/                     # 图形界面与后台线程
requirements.txt             # Python 依赖
environment.yml              # Conda 环境配置
data/demo/bus.jpg            # 示例图片
models/                      # 本地模型权重
docs/                        # 课程报告、配图、实验摘要与结项说明
outputs/detections/          # 界面保存的检测结果
```

`models/` 和 `outputs/detections/` 在下载权重、保存结果时按需创建。公开仓库提供应用源码、示例图片、依赖配置与报告材料。

开发工作区另保留 `scripts/`、`tests/`、开发记录及 `outputs/experiments/course_coco24_20261022/` 完整实验材料，通过 `.gitignore` 排除。仅完整开发工作区可使用 `python -B -m unittest discover -s tests -v`，GitHub 克隆不包含这些测试和工具。公开实验摘要记录本次实验的固定样本、协议和统计结果。

## 验证与适用范围

2026-10-22 在现有 macOS arm64 / Python 3.11 环境中重跑 62 项本地回归测试，全部通过；真实 CPU → Qt 后台线程 → 结果接收 → PNG 保存验证通过，示例图使用 `person, bus` 检出 5 个目标。

课程实验使用 COCO val2017 的固定 24 张图片，完成 120 组检测和 30 次预热后耗时测量；默认阈值下 Precision 为 55.46%、Recall 为 50.77%、F1 为 53.01%。这些是选定小样本、预设词汇和固定匹配规则下的结果，不代表 COCO 全集 mAP 或未见类别泛化能力。

Windows/Linux、MPS/CUDA 和全新环境安装尚未完成实机验证。当前交付形式为 Python 源码项目，需要自行配置环境；首次权重下载需要网络。

## 模型来源

模型使用说明见 [Ultralytics YOLO-World 文档](https://docs.ultralytics.com/models/yolo-world/)，使用条款见 [Ultralytics 许可说明](https://www.ultralytics.com/license)。本项目使用上游发布的预训练权重。
