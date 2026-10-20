"""命令行与 GUI 共用的设备选择；导入本模块不加载 PyTorch。"""


def resolve_device(requested):
    if requested not in ("auto", "cpu", "mps", "cuda"):
        raise ValueError(f"不支持的设备：{requested}")
    if requested != "auto":
        return requested

    import torch

    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"
