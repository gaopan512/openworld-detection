import cv2
import hashlib
from src.image_io import read_image

def get_class_color(class_name):
    """根据类别名称生成稳定的BGR颜色"""

    digest = hashlib.md5(class_name.encode("utf-8")).digest()

    b = 55 + digest[0] % 201
    g = 55 + digest[1] % 201
    r = 55 + digest[2] % 201

    return (b, g, r)


def draw_detections(image_path,detections):
    """在图片上绘制目标检测结果"""

    image = read_image(image_path)
    #遍历所有检测结果
    for detection in detections:

        #获取目标边界框坐标
        x1, y1, x2, y2 = map(int ,detection["bbox"])
        #获取目标类别
        class_name = detection["class_name"]
    
        #获取当前类别对应的颜色
        color = get_class_color(class_name)
        #绘制目标检测框
        cv2.rectangle(
            image,
            (x1, y1),
            (x2, y2),
            color,
            2
        )
        #获取目标置信度
       
        confidence = detection["confidence"]
        # 生成标签
        label = f"{class_name} {confidence:.0%}"

        # 获取图片尺寸
        img_height, img_width = image.shape[:2]

        # 设置文字参数
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2
        padding = 5

        # 计算文字允许使用的最大宽度
        max_text_width = max(0, img_width - 2 * padding)

        # 计算原始标签尺寸
        (text_width, text_height), baseline = cv2.getTextSize(
            label, font, font_scale, thickness
        )

        # 仅缩短类别名称，始终为置信度预留宽度；二分查找避免逐字截断。
        if text_width > max_text_width:
            score = f"{confidence:.0%}"
            def fits(text):
                return cv2.getTextSize(text, font, font_scale, thickness)[0][0] <= max_text_width

            if not fits(score):
                label = ""
            elif not fits(f"... {score}"):
                label = score
            else:
                low, high = 0, len(class_name)
                while low < high:
                    middle = (low + high + 1) // 2
                    if fits(f"{class_name[:middle]}... {score}"):
                        low = middle
                    else:
                        high = middle - 1
                label = f"{class_name[:low]}... {score}"
        # 标签无法显示时，保留检测框
        if not label:
            continue

        # 重新计算最终标签尺寸
        (text_width, text_height), baseline = cv2.getTextSize(
            label, font, font_scale, thickness
        )

        box_width = text_width + 2 * padding
        box_height = text_height + baseline + 2 * padding
        if box_height > img_height:
            continue

        # 避免标签超出水平边界
        label_x = max(0, min(x1, img_width - box_width))

        # 优先在检测框上方显示
        if y1 >= box_height:
            label_y = y1 - box_height
        else:
            label_y = max(0, min(y1 + 2, img_height - box_height))

        # 绘制标签背景
        cv2.rectangle(
            image,
            (label_x, label_y),
            (label_x + box_width, label_y + box_height),
            color,
            -1
        )

        # 根据背景亮度选择文字颜色
        b, g, r = color

        brightness = (
            0.114 * b +
            0.587 * g +
            0.299 * r
        )

        if brightness >= 150:
            text_color = (0, 0, 0)
        else:
            text_color = (255, 255, 255)

        # 绘制标签文字
        cv2.putText(
            image,
            label,
            (label_x + padding, label_y + padding + text_height),
            font,
            font_scale,
            text_color,
            thickness
        )
        

    return image
