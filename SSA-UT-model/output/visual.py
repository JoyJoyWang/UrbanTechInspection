import json
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import cv2
import os
from pycocotools import mask as maskUtils
import numpy as np

# 加载 JSON 文件
with open('pan-4k-02661d8eb4b944e798ab0bcb18d7cffb-skybox0_semantic.json', 'r') as json_file:
    loaded_anns = json.load(json_file)

# 使用 OpenCV 加载图像并转换为 RGB
image_path = 'pan-4k-02661d8eb4b944e798ab0bcb18d7cffb-skybox0.jpg'
if not os.path.exists(image_path):
    raise FileNotFoundError(f"Image file not found at path: {image_path}")

img = cv2.imread(image_path)
if img is None:
    raise ValueError(f"Failed to load image file at path: {image_path}")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# 创建一个图像来可视化注释
fig = make_subplots(rows=1, cols=1)
fig.add_trace(go.Image(z=img))

# 遍历注释并绘制掩码和类别名称
for ann in loaded_anns['annotations']:
    mask = maskUtils.decode(ann['segmentation'])
    class_name = ann['class_name']

    # 找到掩码的轮廓
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        contour = np.squeeze(contour, axis=1)  # 去掉不必要的维度

        # 将hovertext转换为字符串格式
        hovertext = (
            f'Class: {class_name}<br>'
            f'Area: {ann["area"]}<br>'
            f'Predicted IOU: {ann["predicted_iou"]}<br>'
            f'Stability Score: {ann["stability_score"]}<br>'
            f'Class Proposals: {", ".join(ann["class_proposals"])}<br>'
        )

        # 绘制轮廓
        fig.add_trace(go.Scatter(
            x=contour[:, 0],
            y=contour[:, 1],
            mode='lines',
            fill='toself',
            line=dict(color='Red'),
            fillcolor='rgba(255,0,0,0.3)',  # 设置填充颜色为红色半透明
            hoverinfo='text',
            hovertext=hovertext,
            name=", ".join(ann["class_proposals"]),  # 设置图例名称为 class proposals
        ))

    # 添加类别名称
    # 找到掩码的中心点
    moments = cv2.moments(mask)
    if moments['m00'] != 0:
        cx = int(moments['m10'] / moments['m00'])
        cy = int(moments['m01'] / moments['m00'])
    else:
        cx, cy = 0, 0

    fig.add_trace(go.Scatter(
        x=[cx],
        y=[cy],
        text=[class_name],
        mode="text",
        textposition="top left",
        textfont=dict(color="Red", size=12),
        hoverinfo='none',  # 禁用 hover 信息
    ))

# 更新布局
fig.update_layout(
    height=img.shape[0],
    width=img.shape[1],
    margin=dict(l=0, r=0, t=0, b=0),
    showlegend=True  # 显示图例
)

# 将图表保存为 HTML 文件
html_output_path = 'annotations_visualization.html'
fig.write_html(html_output_path)
print(f'Interactive visualization saved to {html_output_path}')

# 打开保存的 HTML 文件
import webbrowser

webbrowser.open(f'file://{os.path.abspath(html_output_path)}')
