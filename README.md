# 基于 YOLO-World 的开放词汇目标检测与可视化分析系统

使用 YOLO-World v2 与 PyQt6，对用户输入的英文类别短语进行单张图片目标检测，并显示和保存带检测框的结果图片。

## 当前功能

- 选择或拖入 JPG、JPEG、PNG、BMP 图片，原图与结果通过页签切换。
- 自由输入英文类别短语，无固定类别白名单，不进行中文翻译。
- 调节置信度阈值，使用后台线程加载模型、执行检测和绘图。
- 显示类别标签、百分比置信度和检测框，支持滚轮缩放、拖动平移、双击恢复适应窗口。
- 输入自动生成彩色类别标签；点击标签高亮该类别，虚线表示本次阈值下未检出。
- 可选“输入停顿后自动检测”，默认关闭；开启后修改输入或阈值，停顿约 0.9 秒执行一次真实推理。
- 保存完整分辨率的 PNG/JPEG 结果图片。

允许输入某个类别，不代表模型一定能识别它。提供目标数量、检测明细、类别数量、检测耗时和设备信息；当前不包含视频和摄像头功能。

## 启动

以下方式使用已配置的本地 Conda 环境：

```bash
conda activate openworld-detection
cd /Users/gaopan/Documents/openworld-detection
python -B main.py
```

也可以从项目根目录执行：

```bash
python -B -m src.gui.main_window
```

默认模型位于 `models/yolov8s-worldv2.pt`。权重文件未纳入 Git；首次检测时在后台自动下载缺失或校验不通过的默认 YOLO 和 CLIP 权重，并显示进度，之后复用本地文件与检测器。相同提示词下只改阈值不会重复设置类别。其他本地 `.pt` 模型仍可在下拉框中选择，由用户自行准备，不套用默认权重校验值。

GUI 默认按 MPS、CUDA、CPU 顺序选择可用设备，也可手动指定。自动选择加速设备后，若模型加载或推理失败，会显示原因并用 CPU 重试一次；CPU 再次失败则报告两次错误。下载失败直接报告下载错误，不触发设备回退。手动指定设备时保持报错，不回退。结果中显示最终设备，回退原因保留在状态栏和参数提示中。

## 新环境安装（Mac、Windows、Linux）

在项目根目录执行，建议使用 Conda 提供独立的 Python 3.11 环境：

```bash
conda env create -f environment.yml
conda activate openworld-detection
python -B main.py
```

已有独立 Python 3.11 环境时，也可执行 `python -m pip install -r requirements.txt`。安装 CLIP 的固定 Git 提交需要 Git 和网络。默认权重由程序在首次检测时获取，也可以先在项目根目录执行：

```bash
python -B -m src.model_assets
python -B main.py
```

两份权重合计约 363 MiB；具体下载、缓存和离线准备规则见下文。

`requirements.txt` 记录当前 Mac 中直接依赖的已安装版本与 CLIP 提交，`environment.yml` 提供环境创建入口；这不是所有间接依赖的完整锁定，也尚未在全新环境及 Windows/Linux 实机安装验证。Windows 的时区数据库通过 `tzdata` 补齐。CUDA 需要兼容的显卡、驱动及支持 CUDA 的 PyTorch；遇到平台无对应 wheel 或驱动不兼容时，应按目标平台调整 PyTorch 安装组合，不能以当前 Mac 的版本记录代替实机验证。Linux 桌面还需可用的显示服务及 Qt 系统依赖；无显示环境仅运行自动测试和无窗口验证。

Mac 可选择 MPS 或 CPU，Windows/Linux 可选择 CUDA 或 CPU。图片读写通过字节编解码兼容中文路径；EXIF 方向在原图预览与检测绘图中保持一致。

## 默认权重下载与离线准备

- YOLO：固定使用 [Ultralytics 官方 v8.3.0 Release](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8s-worldv2.pt)，25,923,032 字节，保存到 `models/yolov8s-worldv2.pt`。
- CLIP：使用固定 CLIP 依赖中的 [ViT-B/32 官方地址](https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt)，353,976,522 字节，保存到当前 Ultralytics 的 `WEIGHTS_DIR/clip/ViT-B-32.pt`。预下载命令会打印两份文件的绝对位置；CLIP 缓存通常位于项目目录外，与现有 Ultralytics 行为一致。
- 两份文件均检查固定大小和 SHA-256；已校验的本地文件不联网。版本与校验值记录在 `src/model_assets.py`，不会追随 latest 自动升级。
- 每份文件最多尝试 3 次，连接/读取等待超时为 20 秒，失败后间隔 1、2 秒重试。每次从头下载；临时文件与目标在同一目录，只有完整校验通过后才原子替换目标。普通失败会清理临时文件，保留原有目标文件。强制结束进程可能遗留隐藏的 `.part` 文件，但不会作为模型加载。
- 网络失败后可修复网络，再点击检测或重跑预下载命令。也可按错误中给出的官方地址和目标位置手动放置文件，再运行命令校验。不会绕过校验加载损坏的默认权重。
- 离线使用前，应在联网环境完成依赖安装和两份权重准备；只复制 YOLO 权重不足以支持自由提示词检测。自定义模型的额外依赖由其加载逻辑处理。

模型来源与使用条款见 [Ultralytics YOLO-World 文档](https://docs.ultralytics.com/models/yolo-world/) 和 [Ultralytics 许可说明](https://www.ultralytics.com/license)。此项目直接从上游下载权重，不重新托管，也不改变上游许可。

## 操作步骤

1. 点击“打开图片”，可先使用 `data/demo/bus.jpg`。
2. 输入英文检测目标，例如 `person, bus, traffic light`。
3. 设置置信度阈值，默认 `0.25`。
4. 点击“开始检测”，等待结果显示。检测期间不能更换图片或重复提交，也需等待检测结束后再关闭窗口。
5. 点击“保存结果”，选择文件名、位置和格式。

标签边框颜色与检测框一致。点击明细表中的一行，可高亮对应目标框。类别标签可高亮同类目标；再次点击同一标签取消高亮，高亮只影响预览，不写入保存图片。更改输入后，旧结果及其原参数仍可查看与保存，标签显示“待检测”；开始新任务后旧结果清空，失败时清除旧参数说明。自动检测是一次次完整推理，不是对旧结果单纯筛选。打开图片或保存结果期间暂停待执行的自动检测，对话框关闭后恢复；保存使用进入对话框之前的结果快照。

长类别标签优先缩短类别名并保留置信度；图片过窄或过矮而无法放下标签时保留检测框。检测耗时包含权重校验与可能发生的下载、模型加载、文本编码、推理及可能发生的 CPU 重试，不包含绘图和保存；不作为纯推理速度基准。

首次推理可能需要额外初始化时间。未检测到目标时仍可保存原图对应的结果；推理失败会弹出错误信息，结束后可再次检测。

## 提示词规则

- 使用逗号分隔不同类别，兼容中文逗号。
- 保留多词短语，例如 `fire extinguisher`。
- 清理多余空白，忽略空项；按忽略大小写去重，保留首次写法和顺序。
- 空输入或包含中文汉字时提示错误。
- 解析器只整理文本，不验证短语是否确为英文物体名称，也不保证模型能识别。

## 保存位置

检测完成后不会自动保存；点击“保存结果”后，默认目录为项目中的：

```text
outputs/detections/
```

默认文件名是 `原图名_detected.png`，例如 `bus_detected.png`。可在保存窗口中选择其他位置，或使用 JPEG 格式。保存使用完整分辨率的结果图片，不使用缩小后的窗口预览。

程序会阻止覆盖当前原图，覆盖其他已有文件前会确认。`integration_test.jpg` 是集成测试输出，与 GUI 保存结果分别命名。生成的检测图片已由 `.gitignore` 忽略。

## 主要文件

```text
main.py                              # 项目启动入口
src/class_parser.py                  # 英文提示词解析
src/detector.py                      # 模型加载与目标检测
src/device.py                        # 跨平台设备选择
src/model_assets.py                  # 默认 YOLO/CLIP 权重下载与校验
src/image_io.py                      # 支持中文路径的图片读写
src/visualizer.py                    # 检测框和标签绘制
src/gui/main_window.py               # 窗口、预览、输入与保存
src/gui/worker.py                    # 后台检测线程
src/gui/style.py                     # 浅色界面样式
scripts/test_class_parser.py         # 提示词解析测试
scripts/test_detection_visualization.py  # 交互式检测与绘图验证
scripts/run_prompt_experiments.py    # 真实提示词与阈值对比实验
tests/test_gui.py                    # GUI 与后台线程回归测试
scripts/verify_real_inference.py      # 真实模型到 GUI 结果的无窗口验证
requirements.txt                     # 当前环境直接依赖的版本记录
environment.yml                      # Conda 环境创建入口
data/demo/bus.jpg                    # 原始测试图片
models/yolov8s-worldv2.pt             # 本地模型权重
outputs/detections/                  # 检测结果图片
```

## 验证与环境记录

提示词解析测试（不加载模型）：

```bash
python -B scripts/test_class_parser.py
```

命令行集成验证（默认 CPU，会生成或覆盖测试结果图片；可追加 `--device mps` 或 `--device cuda`）：

```bash
python -B scripts/test_detection_visualization.py
```

统一回归测试（不弹出窗口，不加载真实权重；使用项目环境中的依赖）：

```bash
python -B -m unittest discover -s tests -v
```

测试覆盖提示词解析、检测器输入校验和类别缓存、设备选择、CPU 回退与失败上限、Qt 后台线程与信号、对话框与自动检测互斥、EXIF 旋转和镜像、长标签、中文路径、保存编码与原图保护。推理使用替代对象，因此这些测试不验证模型识别效果或真实加速设备兼容性；测试文件位于自动清理的临时目录。

真实权重到 Qt 后台线程、GUI 结果接收及 PNG 编码的无窗口验证：

```bash
python -B scripts/verify_real_inference.py --device cpu
```

可追加 `--image 图片路径`、`--classes "person, bus"`，或显式选择 `mps` / `cuda` / `auto`。每次在 `outputs/verification/时间目录/` 保存 `result.png` 与 `result.json`，后者包含最终设备、回退原因及检测明细。该脚本不验证原生文件对话框的手动操作和桌面显示效果，也不计算识别准确率。

## 真实模型对比实验

在同一张图片上执行五组实验：`person, bus` 的阈值分别为 `0.25`、`0.50`、`0.75`，以及在 `0.25` 下分别替换为 `pedestrian, bus` 和 `person, coach`。每组只改变提示词或阈值中的一项。脚本中的类别是实验条件，不限制 GUI 自由输入。

```bash
python -B scripts/run_prompt_experiments.py --device cpu
```

也可通过 `--image 图片路径` 指定另一张图片，或在支持的环境中使用 `--device mps` / `--device cuda`。实验脚本固定使用指定设备，不自动回退，以免混合实验条件。

每次实验使用独立时间目录，时区为 Asia/Shanghai：

- `outputs/detections/experiments/时间目录/`：五组带框结果图片。
- `outputs/experiments/时间目录/results.json`：输入图片和权重的 SHA-256、环境版本、实际设备、提示词、阈值、检测框、类别与置信度。实验失败时保留已完成记录，并标记失败状态。

这些实验用于观察提示词和阈值对结果的影响。没有人工真值标注时，不计算准确率、召回率或 mAP，也不根据单张图片判断哪种提示词普遍更好。本脚本不做速度基准，不新增 GUI 统计功能。

2026-10-18 已核对的本地安装版本：Python 3.11.17、Ultralytics 8.4.177、PyTorch 2.14.1、OpenCV 5.0.0.93、PyQt6 6.11.0。这是现有环境记录，不是已验证的全新环境安装方案。

真实 CPU 推理到 GUI 结果传递已通过本地无窗口验证，示例 `bus.jpg`、提示词 `person, bus`、阈值 0.25 输出 5 个目标，结果为 810 × 1080。记录保存在 `outputs/verification/20261019_231301_857473/`。2026-10-20 用户确认 macOS 桌面验收通过：图片打开与取消、CPU 检测、原图/结果切换、缩放拖动、高亮、自动检测及 PNG/JPEG 保存。桌面验收由用户实际操作确认，自动化桌面工具未能连接 Qt 窗口。当前验证进程的 MPS 与 CUDA 均不可用；Windows/Linux 实机、全新环境安装和真实加速推理尚未验证。真实模型验证脚本现已保存为 `scripts/verify_real_inference.py`。

## GUI 功能恢复验证（2026-10-19）

恢复默认关闭的 900 毫秒自动检测、滚轮缩放与拖动平移、双击适应窗口，以及随自由输入生成的类别标签（含待检测、未检出和类别高亮）。保留检测明细表联动和类别统计。自动检测对有效输入、阈值、模型、设备和新图片的变化重新计时；关闭窗口或手动开始检测时取消待触发任务，推理期间锁定参数。高亮仅为预览叠加层，保存仍使用完整结果图片。
