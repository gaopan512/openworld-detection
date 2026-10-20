import re


def parse_classes(text):
    """将英文类别提示词解析为有序、去重的类别列表。"""

    # 检查输入类型
    if not isinstance(text, str):
        raise TypeError("检测提示词必须是字符串")

    # 当前版本不处理中文，也不进行中文翻译
    if re.search(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff"
                 r"\U00020000-\U000323af]", text):
        raise ValueError("请使用英文类别短语，当前版本不支持中文输入")

    # 将中文逗号视为分隔符，方便输入
    text = text.replace("，", ",")

    classes = []
    seen = set()

    for item in text.split(","):
        # 去掉首尾空白，并将内部连续空白合并为一个空格
        class_name = " ".join(item.split())

        # 忽略空项
        if not class_name:
            continue

        # 忽略大小写判断重复，保留第一次输入的写法
        key = class_name.casefold()

        if key in seen:
            continue

        seen.add(key)
        classes.append(class_name)

    if not classes:
        raise ValueError("请至少输入一个英文类别短语")

    return classes