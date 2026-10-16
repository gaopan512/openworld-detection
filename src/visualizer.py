import cv2

def draw_detections(image_path,detections):
    """在图片上绘制目标检测结果"""

    image = cv2.imread(str(image_path))

    if image is None:
        raise ValueError(f"无法读取图片：{image_path}")
    #不同类别对应不同颜色
    colors = {
        "person":(0,255,0),
        "car":(255,0,0),
        "bicycle":(0,255,255),
        "dog":(0,0,255)
        }
    #遍历所有检测结果
    for detection in detections:

        #获取目标边界框坐标
        x1, y1, x2, y2 = map(int ,detection["bbox"])
        #获取目标类别
        class_name = detection["class_name"]
    
        #获取当前类别对应的颜色
        color = colors.get(class_name, (255,255,255))
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
        #生成显示标签
        label=f"{class_name} {confidence:.2f}"

        #在检测框上方绘制文字
        cv2.putText(
            image,
            label,
            (x1, max(y1 - 10,20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2
        )
    return image